import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import math

"""
二维SN中子输运方程求解器
功能：
1. 支持 S2, S4, S8, S12 阶数
2. 自动归一化权重，防止发散
3. 自动绘制热力图和3D图
"""

class SNSolver2D:
    def __init__(self, N, sigma_t, sigma_s0, S0_strength, mesh_size=2.0):
        self.N = N 
        self.sigma_t = sigma_t
        self.sigma_s0 = sigma_s0
        self.S0_strength = S0_strength

        # 几何定义: 100cm x 100cm
        self.L_x = 100.0
        self.L_y = 100.0
        self.dx = mesh_size
        self.dy = mesh_size
        self.nx = int(self.L_x / self.dx)
        self.ny = int(self.L_y / self.dy)
        
        # 初始化通量
        self.phi = np.zeros((self.ny, self.nx)) 
        
        # 设置源分布
        self.Q_exist = np.zeros((self.ny, self.nx))
        self._setup_source()
        
        # 获取角度求积组 (自动修复权重的版本)
        self.quads = self._get_quadrature(N)  
        
    def _setup_source(self):
        """定义源区域: 0 < x < 25, 25 < y < 50"""
        ix_end = int(25.0 / self.dx)      
        iy_start = int(25.0 / self.dy)
        iy_end = int(50.0 / self.dy)
        self.Q_exist[iy_start:iy_end, 0:ix_end] = self.S0_strength

    def _get_quadrature(self, N):
        """
        通用 SN 求积组生成器 (Level Symmetric Quadrature)
        关键修正：增加自动权重归一化，防止计算发散。
        """
        quads = []
        octant_data = [] # 存储第一卦限的点
        
        # === 1. 定义标准参数 ===
        mu_sq = []
        w_data = [] 

        if N == 2:
            # S2: 简单处理
            mu = 0.5773502692
            w = np.pi # 4个点，总和4pi
            for mx in [mu, -mu]:
                for my in [mu, -mu]:
                    quads.append({'mu': mx, 'eta': my, 'w': w})
            return quads

        elif N == 4:
            mu_sq = [0.3500212**2, 0.8688903**2]
            w_data = [0.3333333333] 

        elif N == 8:
            mu_sq = [0.04092416, 0.16368630, 0.36732410, 0.65306120]
            w_data = [0.05719086, 0.04641666, 0.04470377]

        elif N == 12:
            mu_sq = [
                0.01826563, 0.07340081, 0.16521360, 
                0.29369400, 0.45884270, 0.66068210
            ]
            w_data = [
                0.02677465, 0.02235921, 0.02058897, 
                0.02055611, 0.01917637, 0.01887372
            ]
        else:
            raise NotImplementedError(f"S{N} not implemented.")

        # === 2. 生成第一卦限点 ===
        self._generate_octant_data(N, mu_sq, w_data, octant_data)

        # === 3. 镜像到4个象限 (暂不缩放权重) ===
        raw_quads = []
        for p in octant_data:
            # 扩展到4个象限
            for sx in [1, -1]:
                for sy in [1, -1]:
                    # 注意：这里2D方向其实代表了3D的上下两个方向
                    # 但我们先不管系数，最后统一拉伸
                    raw_quads.append({
                        'mu': p['mu'] * sx,
                        'eta': p['eta'] * sy,
                        'w': p['w'] 
                    })
        
        # === 4. 【关键】强制归一化到 4pi ===
        current_sum = sum([q['w'] for q in raw_quads])
        target_sum = 4.0 * np.pi
        scale_factor = target_sum / current_sum
        
        # print(f"  [Info] S{N} Weight Scaling: RawSum={current_sum:.4f} -> Target={target_sum:.4f}")
        
        for q in raw_quads:
            q['w'] *= scale_factor
            quads.append(q)
            
        return quads

    def _generate_octant_data(self, N, mu_sq_list, w_list, target_list):
        """辅助函数：生成第一卦限数据"""
        M = N // 2 
        mu_vals = [math.sqrt(x) for x in mu_sq_list]
        idx_w = 0
        
        # 标准 LSQ 遍历顺序
        for i in range(M):
            for j in range(M - i):
                mu = mu_vals[i]
                eta = mu_vals[j]
                w = w_list[idx_w] if idx_w < len(w_list) else w_list[-1]
                target_list.append({'mu': mu, 'eta': eta, 'w': w})
                idx_w += 1

    def solve(self, max_iter=2000, tol=1e-4):
        print(f"Starting S{self.N} Calculation ({len(self.quads)} angles)...")
        start_time = time.time()
        
        for k in range(max_iter):
            phi_old = self.phi.copy()
            phi_new = np.zeros_like(self.phi)
            
            # 源项计算：(Scattering + Fixed) / 4pi
            source_term = (self.sigma_s0 * phi_old + self.Q_exist) / (4.0 * np.pi) 
            
            # 角度扫描
            for quad in self.quads: 
                mu = quad['mu']
                eta = quad['eta']
                w = quad['w']
                
                # 确定扫描方向和范围
                if mu > 0:
                    x_range = range(self.nx)
                    dx_eff = self.dx
                else:
                    x_range = range(self.nx - 1, -1, -1)
                    dx_eff = self.dx
                
                if eta > 0:
                    y_range = range(self.ny)
                    dy_eff = self.dy
                else:
                    y_range = range(self.ny - 1, -1, -1)
                    dy_eff = self.dy
                
                # 边界条件缓冲 (Rolling Buffers)
                psi_in_x = np.zeros(self.ny) # 行入流
                psi_in_y = np.zeros(self.nx) # 列入流
                
                # 预计算常数
                two_mu_dx = 2.0 * abs(mu) / dx_eff
                two_eta_dy = 2.0 * abs(eta) / dy_eff
                denom = self.sigma_t + two_mu_dx + two_eta_dy
                
                # 空间扫描 Loop
                for iy in y_range:
                    psi_x_prev = psi_in_x[iy] # 当前行的入流
                    row_source = source_term[iy, :]
                    
                    for ix in x_range:
                        Q = row_source[ix]
                        psi_y_prev = psi_in_y[ix] # 当前列的入流
                        
                        # 菱形差分公式
                        psi_center = (Q + two_mu_dx * psi_x_prev + two_eta_dy * psi_y_prev) / denom
                        
                        # 负通量修正
                        if psi_center < 0: psi_center = 0.0
                        
                        # 累加标量通量
                        phi_new[iy, ix] += w * psi_center
                        
                        # 递推边界
                        psi_x_out = 2.0 * psi_center - psi_x_prev
                        psi_y_out = 2.0 * psi_center - psi_y_prev
                        
                        # 更新缓冲
                        psi_x_prev = psi_x_out
                        psi_in_y[ix] = psi_y_out
                    
                    psi_in_x[iy] = psi_x_prev # 保存行出流

            # 收敛检查
            max_val = np.max(phi_new)
            if max_val > 1e-20:
                rel_err = np.max(np.abs(phi_new - phi_old)) / max_val
            else:
                rel_err = 1.0
                
            self.phi = phi_new
            
            if rel_err < tol:
                print(f"  -> Converged at iter {k}. Time: {time.time()-start_time:.2f}s. Max Flux: {max_val:.4e}")
                break
        else:
            print(f"  -> Reached max iter {max_iter}. Rel Err: {rel_err:.4e}")
            
        return self.phi

    def plot_heatmap(self):
        """绘制 2D 热力图"""
        plt.figure(figsize=(7, 6))
        extent = [0, self.L_x, 0, self.L_y]
        plt.imshow(self.phi, origin='lower', extent=extent, cmap='jet')
        plt.colorbar(label=r'Scalar Flux $\phi$')
        # 画源框
        plt.plot([0, 25, 25, 0, 0], [25, 25, 50, 50, 25], 'w--', linewidth=1.5, label='Source')
        plt.title(f'Scalar Flux Distribution ($S_{{{self.N}}}$)')
        plt.xlabel('x (cm)')
        plt.ylabel('y (cm)')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    def plot_3d(self):
        """绘制 3D 表面图"""
        x = np.linspace(0, self.L_x, self.nx)
        y = np.linspace(0, self.L_y, self.ny)
        X, Y = np.meshgrid(x, y)
        
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, self.phi, cmap='jet', 
                               edgecolor='none', antialiased=True, rstride=1, cstride=1)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=r'Flux $\phi$')
        
        ax.set_title(f'3D Flux Landscape ($S_{{{self.N}}}$)')
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Flux')
        # 调整视角看清源头
        ax.view_init(elev=35, azim=230)
        plt.tight_layout()
        plt.show()

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 题目物理参数
    Sigma_t = 0.25
    Sigma_s0 = 0.15
    # 题目给的 Sigma_s1, s2 在各向同性近似下暂不使用，若需高阶需改源项公式
    S0 = 1e14
    
    # 我们要跑的 SN 阶数列表
    sn_orders = [2, 4, 8, 12]
    
    for n in sn_orders:
        print(f"\n{'='*30}")
        print(f" Running S{n} Approximation")
        print(f"{'='*30}")
        
        # 1. 初始化求解器
        solver = SNSolver2D(N=n, sigma_t=Sigma_t, sigma_s0=Sigma_s0, S0_strength=S0, mesh_size=2.0)
        
        # 2. 求解
        solver.solve()
        
        # 3. 画热力图
        print(f"Displaying Heatmap for S{n}...")
        solver.plot_heatmap()
        
        # 4. 画3D图
        print(f"Displaying 3D Plot for S{n}...")
        solver.plot_3d()