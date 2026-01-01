import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

class MOCFixedSource:
    def __init__(self, mesh_size=1.0, num_angles=32):
        # === 1. 几何与网格 ===
        self.L_x = 100.0
        self.L_y = 100.0
        self.dx = mesh_size
        self.dy = mesh_size
        self.nx = int(self.L_x / self.dx)
        self.ny = int(self.L_y / self.dy)
        self.mesh_area = self.dx * self.dy
        
        # === 2. 物理参数 ===
        # 吸收截面 Sigma_a = 0.25 - 0.15 = 0.10
        # 理论估算通量 = S / Sigma_a = 1e14 / 0.1 = 1e15 左右
        self.sigma_t = 0.25
        self.sigma_s = 0.15
        self.S0_strength = 1e14
        
        # === 3. 初始化场量 ===
        self.phi = np.zeros((self.ny, self.nx))
        self.fixed_source = np.zeros((self.ny, self.nx))
        self.total_source = np.zeros((self.ny, self.nx))
        
        # 设置固定源区域 (25 < x < 50 修正为左下角区域以匹配SN)
        # 源区域: x < 25, 25 < y < 50 (根据之前的设定)
        ix_end = int(25.0 / self.dx)
        iy_start = int(25.0 / self.dy)
        iy_end = int(50.0 / self.dy)
        self.fixed_source[iy_start:iy_end, 0:ix_end] = self.S0_strength
        
        # === 4. 角度设置 ===
        self.num_angles = num_angles
        self.phis = np.linspace(0.01, 2*np.pi-0.01, num_angles) # 避开完全的0度90度
        self.weights = np.ones(num_angles) / num_angles 

    def trace_ray(self, x0, y0, phi):
        """射线追踪"""
        tracks = []
        x, y = x0, y0
        sin_p = np.sin(phi)
        cos_p = np.cos(phi)
        
        # 极小值保护
        if abs(cos_p) < 1e-6: cos_p = 1e-6 if cos_p >= 0 else -1e-6
        if abs(sin_p) < 1e-6: sin_p = 1e-6 if sin_p >= 0 else -1e-6
        
        step_x = 1 if cos_p > 0 else -1
        step_y = 1 if sin_p > 0 else -1
        
        idx_x = int(x // self.dx)
        idx_y = int(y // self.dy)
        idx_x = max(0, min(idx_x, self.nx - 1))
        idx_y = max(0, min(idx_y, self.ny - 1))

        while 0 <= idx_x < self.nx and 0 <= idx_y < self.ny:
            if step_x > 0:
                d_to_x = ((idx_x + 1) * self.dx - x) / cos_p
            else:
                d_to_x = (idx_x * self.dx - x) / cos_p
                
            if step_y > 0:
                d_to_y = ((idx_y + 1) * self.dy - y) / sin_p
            else:
                d_to_y = (idx_y * self.dy - y) / sin_p
            
            dist = min(d_to_x, d_to_y)
            tracks.append((dist, idx_x, idx_y))
            
            x += dist * cos_p
            y += dist * sin_p
            
            if abs(dist - d_to_x) < 1e-8:
                idx_x += step_x
            else:
                idx_y += step_y
                
        return tracks

    def transport_sweep(self):
        """修正后的扫描：增加投影修正 (Projection Correction)"""
        new_phi_accum = np.zeros((self.ny, self.nx))
        
        ray_spacing = 0.5 # 间距
        
        for ang_idx, phi in enumerate(self.phis):
            w = self.weights[ang_idx]
            sin_p = np.sin(phi)
            cos_p = np.cos(phi)
            
            rays = []
            
            # === 1. 垂直边界发射 (Left/Right) ===
            # 有效宽度 delta_r = spacing * |cos(phi)|
            delta_r_vert = ray_spacing * abs(cos_p)
            
            if cos_p > 0: # From Left
                ys = np.arange(0, self.L_y, ray_spacing)
                for y in ys: rays.append((0, y + 1e-4, delta_r_vert))
            else: # From Right
                ys = np.arange(0, self.L_y, ray_spacing)
                for y in ys: rays.append((self.L_x, y + 1e-4, delta_r_vert))
            
            # === 2. 水平边界发射 (Bottom/Top) ===
            # 有效宽度 delta_r = spacing * |sin(phi)|
            delta_r_horiz = ray_spacing * abs(sin_p)
            
            if sin_p > 0: # From Bottom
                xs = np.arange(0, self.L_x, ray_spacing)
                for x in xs: rays.append((x + 1e-4, 0, delta_r_horiz))
            else: # From Top
                xs = np.arange(0, self.L_x, ray_spacing)
                for x in xs: rays.append((x + 1e-4, self.L_y, delta_r_horiz))

            # === 3. 追踪所有射线 ===
            for (rx, ry, eff_width) in rays:
                psi_in = 0.0
                tracks = self.trace_ray(rx, ry, phi)
                
                for (s, ix, iy) in tracks:
                    if s < 1e-10: continue
                    
                    st = self.sigma_t
                    q_iso = self.total_source[iy, ix] / (4 * np.pi)
                    
                    exp_val = np.exp(-st * s)
                    psi_out = psi_in * exp_val + (q_iso / st) * (1.0 - exp_val)
                    psi_avg = (q_iso / st) - (psi_out - psi_in) / (st * s)
                    
                    # 【核心修正点】
                    # 贡献 = psi_avg * 角度权重 * 有效宽度 * 线段长度
                    contribution = psi_avg * w * eff_width * s
                    new_phi_accum[iy, ix] += contribution
                    
                    psi_in = psi_out

        # 归一化: Phi = (4pi / Area) * Sum(...)
        new_phi = new_phi_accum * (4 * np.pi) / self.mesh_area
        
        return new_phi

    def solve(self, max_iter=50, tol=1e-4):
        print(f"Starting MOC Fixed Source Calculation (Corrected)...")
        start_time = time.time()
        
        for k in range(max_iter):
            phi_old = self.phi.copy()
            self.total_source = self.fixed_source + self.sigma_s * self.phi
            phi_new = self.transport_sweep()
            
            max_flux = np.max(phi_new)
            if max_flux < 1e-20: max_flux = 1.0
            
            err = np.max(np.abs(phi_new - phi_old)) / max_flux
            self.phi = phi_new
            print(f"  Iter {k+1:02d}: Max Flux = {max_flux:.4e}, Rel Err = {err:.4e}")
            
            if err < tol:
                print(f"Converged in {k+1} iterations. Time: {time.time()-start_time:.2f}s")
                break
                
        return self.phi

    def plot_results(self):
        plt.figure(figsize=(7, 6))
        extent = [0, self.L_x, 0, self.L_y]
        plt.imshow(self.phi, origin='lower', extent=extent, cmap='jet')
        plt.colorbar(label=r'Scalar Flux $\phi$')
        plt.plot([0, 25, 25, 0, 0], [25, 25, 50, 50, 25], 'w--', linewidth=1.5)
        plt.title('MOC Fixed Source Flux (Corrected)')
        plt.xlabel('x (cm)')
        plt.ylabel('y (cm)')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # 增加射线密度 (mesh_size 2.0 -> ray_spacing 0.5)
    # 增加角度数以获得平滑结果
    moc = MOCFixedSource(mesh_size=2.0, num_angles=24)
    moc.solve(max_iter=50)
    moc.plot_results()