import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class RectangularSolver:
    def __init__(self, bounds, n_mesh_x, n_mesh_y):
        self.bounds = bounds 
        self.nx = n_mesh_x
        self.ny = n_mesh_y
        
        # 网格生成
        self.dx = (bounds[1] - bounds[0]) / n_mesh_x
        self.dy = (bounds[3] - bounds[2]) / n_mesh_y
        self.x_centers = np.linspace(bounds[0]+self.dx/2, bounds[1]-self.dx/2, n_mesh_x)
        self.y_centers = np.linspace(bounds[2]+self.dy/2, bounds[3]-self.dy/2, n_mesh_y)
        
        # 物理量 (Group 1=Fast, Group 2=Thermal)
        self.flux = np.ones((n_mesh_x, n_mesh_y, 2)) 
        self.source = np.zeros((n_mesh_x, n_mesh_y, 2))
        self.k_eff = 1.0
        self.k_history = []
        
        self.mat_map = np.zeros((n_mesh_x, n_mesh_y), dtype=int)
        self.materials = {}

    def define_materials(self):
        """
        根据 Table P4.12 转换参数
        Diffusion -> Transport Mapping: Sig_t = 1/(3D)
        """
        # Core 1
        st1_c1 = 1/(3*1.267)
        st2_c1 = 1/(3*0.354)
        # Sigma_s_self = Sigma_t - Sigma_a - Sigma_removal
        ss11_c1 = st1_c1 - 0.0121 - 0.0241
        ss22_c1 = st2_c1 - 0.121
        
        self.materials[1] = {
            'st': [st1_c1, st2_c1],
            'nusf': [0.0085, 0.1851],
            'ss_self': [ss11_c1, ss22_c1],
            'ss_12': 0.0241,
            'chi': [1.0, 0.0]
        }

        # Core 2
        st1_c2 = 1/(3*1.280)
        st2_c2 = 1/(3*0.400)
        ss11_c2 = st1_c2 - 0.010 - 0.016
        ss22_c2 = st2_c2 - 0.100
        
        self.materials[2] = {
            'st': [st1_c2, st2_c2],
            'nusf': [0.006, 0.150],
            'ss_self': [ss11_c2, ss22_c2],
            'ss_12': 0.016,
            'chi': [1.0, 0.0]
        }

        # Reflector
        st1_r = 1/(3*1.130)
        st2_r = 1/(3*0.166)
        ss11_r = st1_r - 0.0004 - 0.0493
        ss22_r = st2_r - 0.020
        
        self.materials[3] = {
            'st': [st1_r, st2_r],
            'nusf': [0.0, 0.0],
            'ss_self': [ss11_r, ss22_r],
            'ss_12': 0.0493,
            'chi': [0.0, 0.0]
        }

    def build_geometry(self):
        # Reflector bounds (Outer): [-65, 65], [0, 120]
        # Core inner bounds: [-50, 50]
        for i in range(self.nx):
            for j in range(self.ny):
                x = self.x_centers[i]
                y = self.y_centers[j]
                
                mat_id = 3 # Reflector default
                
                # Core Region Check
                if -50 < x < 50:
                    if 15 < y < 55:
                        mat_id = 1 # Core 1
                    elif 55 < y < 105:
                        mat_id = 2 # Core 2
                        
                self.mat_map[i, j] = mat_id

    def sn_sweep(self):
        """
        标准 Diamond Difference (DD) S2 扫描
        这是处理矩形网格最稳健的方法
        """
        new_flux = np.zeros_like(self.flux)
        
        # S2 Quadrature (2D projection)
        # 4个方向，每个象限一个
        # mu = dx/ds, eta = dy/ds. Sum(w) = 1.0
        # For 2D isotropy, w=0.25
        # Direction cosines for S2: 1/sqrt(3) for 3D, but usually 0.577 is used
        mu_abs = 0.57735
        eta_abs = 0.57735
        weight = 0.25 * 4 * np.pi # 积分结果直接对应标量通量
        
        directions = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
        
        for mux_sign, muy_sign in directions:
            mu = mux_sign * mu_abs
            eta = muy_sign * eta_abs
            
            # 确定扫描顺序 (Upwind Scheme)
            x_indices = range(self.nx) if mux_sign > 0 else range(self.nx-1, -1, -1)
            y_indices = range(self.ny) if muy_sign > 0 else range(self.ny-1, -1, -1)
            
            # 边界通量缓存 (Boundary Conditions: Vacuum = 0.0)
            psi_x_bound = np.zeros((self.ny, 2)) # Vertical edges
            psi_y_bound = np.zeros((self.nx, 2)) # Horizontal edges
            
            for i in x_indices:
                for j in y_indices:
                    mat_id = self.mat_map[i, j]
                    mat = self.materials[mat_id]
                    
                    # 获取入射通量
                    # 如果是从左往右扫(mu>0)，入射是左边界 psi_x_bound
                    # 此时 i 对应的左边界索引就是 i (因为 psi_bound 是动态更新的buffer)
                    # 我们简化 buffer 逻辑：直接用单个变量传递
                    
                    # 为了逻辑清晰，我们使用面通量数组 (Array Access)
                    # 但为了速度，这里用局部变量 logic
                    psi_in_x = psi_x_bound[j, :] 
                    psi_in_y = psi_y_bound[i, :]
                    
                    for g in range(2):
                        st = mat['st'][g]
                        q_ang = self.source[i, j, g] / (4 * np.pi)
                        
                        # Diamond Difference Equation
                        # Psi_center = (Q + |mu|/dx * Psi_in_x + |eta|/dy * Psi_in_y) / (St + |mu|/dx + |eta|/dy)
                        
                        coeff_x = mu_abs / self.dx
                        coeff_y = eta_abs / self.dy
                        
                        numerator = q_ang + coeff_x * psi_in_x[g] + coeff_y * psi_in_y[g]
                        denominator = st + coeff_x + coeff_y
                        
                        psi_center = numerator / denominator
                        
                        # Extrapolate to Outgoing Edges
                        # Psi_out = 2 * Psi_center - Psi_in
                        psi_out_x = 2 * psi_center - psi_in_x[g]
                        psi_out_y = 2 * psi_center - psi_in_y[g]
                        
                        # Fix Negative Flux (Step Differencing fallback if needed)
                        # 为简单起见，这里仅做截断
                        if psi_out_x < 0: psi_out_x = 0
                        if psi_out_y < 0: psi_out_y = 0
                        
                        # Accumulate Scalar Flux
                        new_flux[i, j, g] += psi_center * weight
                        
                        # Update Boundary Buffers for next cell
                        psi_x_bound[j, g] = psi_out_x
                        psi_y_bound[i, g] = psi_out_y
                        
        return new_flux

    def solve(self, max_iter=500):
        print(">> 初始化几何与材料...")
        self.define_materials()
        self.build_geometry()
        
        print(f">> 开始两能群迭代 (Grid: {self.nx}x{self.ny})...")
        
        for it in range(1, max_iter+1):
            # 1. Update Source
            fiss_src = np.zeros((self.nx, self.ny))
            for i in range(self.nx):
                for j in range(self.ny):
                    mat = self.materials[self.mat_map[i,j]]
                    fiss_src[i,j] = mat['nusf'][0]*self.flux[i,j,0] + mat['nusf'][1]*self.flux[i,j,1]
            
            for i in range(self.nx):
                for j in range(self.ny):
                    mat = self.materials[self.mat_map[i,j]]
                    # G1 Source
                    self.source[i,j,0] = (1.0/self.k_eff)*mat['chi'][0]*fiss_src[i,j] + \
                                         mat['ss_self'][0]*self.flux[i,j,0]
                    # G2 Source (with downscattering)
                    self.source[i,j,1] = (1.0/self.k_eff)*mat['chi'][1]*fiss_src[i,j] + \
                                         mat['ss_self'][1]*self.flux[i,j,1] + \
                                         mat['ss_12']*self.flux[i,j,0]
            
            # 2. Transport Sweep (DD)
            new_flux = self.sn_sweep()
            
            # 3. Update k_eff
            num = 0.0
            den = 0.0
            for i in range(self.nx):
                for j in range(self.ny):
                    mat = self.materials[self.mat_map[i,j]]
                    term_new = mat['nusf'][0]*new_flux[i,j,0] + mat['nusf'][1]*new_flux[i,j,1]
                    term_old = mat['nusf'][0]*self.flux[i,j,0] + mat['nusf'][1]*self.flux[i,j,1]
                    num += term_new
                    den += term_old
            
            k_new = self.k_eff * (num / den)
            self.k_history.append(k_new)
            
            # 4. Check Convergence
            err_k = abs(k_new - self.k_eff) / self.k_eff
            self.k_eff = k_new
            self.flux = new_flux
            
            # Normalize
            self.flux /= np.mean(self.flux)
            
            if it % 10 == 0:
                print(f"Iter {it:3d}: k_eff = {self.k_eff:.6f}, Err = {err_k:.2e}")
            
            if err_k < 1e-5:
                print("\n>> 收敛成功!")
                print(f"   Final k_eff: {self.k_eff:.6f}")
                return self.k_history, self.flux

        return self.k_history, self.flux

# ================= 运行 =================
bounds = [-65, 65, 0, 120]
# 2cm 网格对扩散问题足够
solver = RectangularSolver(bounds, n_mesh_x=65, n_mesh_y=60)
hist, flux = solver.solve()

# ================= 绘图 (修复拼写错误版) =================
flux_g1 = flux[:,:,0].T
flux_g2 = flux[:,:,1].T

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot Fast Flux
im1 = ax1.imshow(flux_g1, origin='lower', extent=bounds, cmap='jet', aspect='auto')
ax1.set_title('Fast Flux (Group 1)')
ax1.set_xlabel('x (cm)') # Fixed typo
ax1.set_ylabel('y (cm)')

# Draw Regions
rect1 = patches.Rectangle((-50, 15), 100, 40, linewidth=1, edgecolor='w', facecolor='none', label='Core 1')
rect2 = patches.Rectangle((-50, 55), 100, 50, linewidth=1, edgecolor='w', facecolor='none', linestyle='--', label='Core 2')
ax1.add_patch(rect1)
ax1.add_patch(rect2)
plt.colorbar(im1, ax=ax1)

# Plot Thermal Flux
im2 = ax2.imshow(flux_g2, origin='lower', extent=bounds, cmap='jet', aspect='auto')
ax2.set_title('Thermal Flux (Group 2)')
ax2.set_xlabel('x (cm)')

rect1b = patches.Rectangle((-50, 15), 100, 40, linewidth=1, edgecolor='w', facecolor='none')
rect2b = patches.Rectangle((-50, 55), 100, 50, linewidth=1, edgecolor='w', facecolor='none', linestyle='--')
ax2.add_patch(rect1b)
ax2.add_patch(rect2b)
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.show()