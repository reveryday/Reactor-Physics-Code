import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import time

# [Group 1, Group 2]
mat_c1 = {'D': [1.267, 0.354], 'Sa': [0.0121, 0.121], 'nSf': [0.0085, 0.1851], 'Ss12': 0.0241}
mat_c2 = {'D': [1.280, 0.400], 'Sa': [0.010, 0.100],  'nSf': [0.006, 0.150],  'Ss12': 0.016}
# Reflector
mat_re = {'D': [1.130, 0.166], 'Sa': [0.0004, 0.020], 'nSf': [0.0, 0.0],      'Ss12': 0.0493}

def plot_3d_fluxes(phi1, phi2, x_coords, y_coords, k_eff):
    """
    绘制快群、热群、总通量的 3D 曲面图
    """
    # 1. 数据准备
    # 创建网格 (Meshgrid)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # 计算总通量
    phi_total = phi1 + phi2
    
    # 为了视觉效果，归一化所有数据 (归一化到总通量的最大值)
    max_val = np.max(phi_total)
    Z1 = phi1 / max_val
    Z2 = phi2 / max_val
    Z_tot = phi_total / max_val

    # 2. 设置绘图布局 (1行3列)
    fig = plt.figure(figsize=(18, 6))
    plt.suptitle(f'3D Neutron Flux Distribution ($k_{{eff}} = {k_eff:.5f}$)', fontsize=16)

    # --- 子函数：画单个 3D 图 ---
    def add_surface_plot(ax, X, Y, Z, title, cmap_name='viridis'):
        # 绘制 3D 曲面
        surf = ax.plot_surface(X, Y, Z, cmap=cmap_name, 
                               linewidth=0, antialiased=False, alpha=0.9)
        
        # 在底部绘制等高线投影 (增加深度感)
        ax.contourf(X, Y, Z, zdir='z', offset=-0.1, cmap=cmap_name, alpha=0.4)
        
        # 设置坐标轴标签
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Normalized Flux')
        ax.set_title(title, fontsize=14, pad=10)
        
        # 设置 Z 轴范围 (留出底部投影空间)
        ax.set_zlim(-0.1, 1.05)
        
        # 调整视角 (Elevation 高度角, Azimuth 方位角)
        ax.view_init(elev=30, azim=-60)
        
        # 添加颜色条
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    # --- 绘制三个图 ---
    
    # (1) 快中子通量 (Fast Flux)
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    add_surface_plot(ax1, X, Y, Z1, r'Fast Flux ($\phi_1$)', cmap_name='autumn')
    # 特点：平滑，像山峰

    # (2) 热中子通量 (Thermal Flux)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    add_surface_plot(ax2, X, Y, Z2, r'Thermal Flux ($\phi_2$)', cmap_name='winter')
    # 特点：注意看反射层边界的“火山口”隆起

    # (3) 总通量 (Total Flux)
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    add_surface_plot(ax3, X, Y, Z_tot, r'Total Flux ($\phi_{tot}$)', cmap_name='jet')

    plt.tight_layout()
    plt.show()


def two_diffusion():
    start_time = time.time()

    # 生成网格
    x_min, x_max = -65.0, 65.0
    y_min, y_max = 0.0, 120.0   
    h = 1.0 # 网格步长 = 1.0 cm   
    nx = int(round((x_max - x_min) / h)) + 1 # 避免浮点数误差
    ny = int(round((y_max - y_min) / h)) + 1
    
    x_coords = np.linspace(x_min, x_max, nx) # 生成x坐标：从-65到 65共131个点
    y_coords = np.linspace(y_min, y_max, ny)
    
    # 构造了7个二维矩阵-材料矩阵
    D1_map = np.zeros((ny, nx)) # 快中子扩散系数矩阵
    D2_map = np.zeros((ny, nx)) # 热中子扩散系数矩阵
    SigR1_map = np.zeros((ny, nx)) # 快中子移出截面矩阵
    SigA2_map = np.zeros((ny, nx)) # 热中子吸收截面矩阵
    nSf1_map = np.zeros((ny, nx)) # 快中子产额截面矩阵
    nSf2_map = np.zeros((ny, nx)) # 热中子产额截面矩阵
    Ss12_map = np.zeros((ny, nx)) # 快到热散射截面矩阵
    
    for j in range(ny):
        y = y_coords[j]
        for i in range(nx):
            x = x_coords[i]
            
            # 判定区域
            in_x_core = -50.0 < x < 50.0
            
            # 注意：这里使用 < 和 <= 需小心，但对于1cm网格，边界恰好落在节点上
            # 我们通常认为节点代表其周围的体积。简单起见，按点坐标判断。
            if in_x_core and (15.0 < y < 55.0):
                m = mat_c1
            elif in_x_core and (55.0 <= y < 105.0):
                m = mat_c2
            else:
                m = mat_re
            
            D1_map[j, i] = m['D'][0]
            D2_map[j, i] = m['D'][1]
            SigR1_map[j, i] = m['Sa'][0] + m['Ss12'] # Fast Removal
            SigA2_map[j, i] = m['Sa'][1]             # Thermal Absorption
            nSf1_map[j, i] = m['nSf'][0]
            nSf2_map[j, i] = m['nSf'][1]
            Ss12_map[j, i] = m['Ss12']

    # ==============================
    # 4. 矩阵构建 (Matrix Assembly)
    # ==============================
    N = nx * ny
    
    def get_idx(j, i):
        return j * nx + i

    def build_diffusion_matrix(D_map, Sigma_rem_map):
        """
        构建五点差分矩阵 (使用 lil_matrix 逐点构建，防止边界错误)
        D 使用调和平均 (Harmonic Mean) 处理界面
        """
        A = lil_matrix((N, N))
        
        for j in range(ny):
            for i in range(nx):
                row = get_idx(j, i)
                
                # --- 边界条件 (Vacuum BC) ---
                # 边界点通量设为 0 -> 方程: 1 * phi = 0
                if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                    A[row, row] = 1.0
                    continue
                
                # --- 内部点 (Internal Nodes) ---
                # D * d2phi/dx2 + Sigma * phi = S
                # 离散化: J_right - J_left + J_up - J_down + Sigma*phi*V = S*V
                # J_right = - D_harm * (phi_i+1 - phi_i) / h
                
                # 1. 计算耦合系数 (Coupling Coefficients = D_harm / h^2)
                # Left (i-1)
                D_L = 2 * D_map[j, i] * D_map[j, i-1] / (D_map[j, i] + D_map[j, i-1])
                coeff_L = D_L / (h**2)
                
                # Right (i+1)
                D_R = 2 * D_map[j, i] * D_map[j, i+1] / (D_map[j, i] + D_map[j, i+1])
                coeff_R = D_R / (h**2)
                
                # Down (j-1)
                D_D = 2 * D_map[j, i] * D_map[j-1, i] / (D_map[j, i] + D_map[j-1, i])
                coeff_D = D_D / (h**2)
                
                # Up (j+1)
                D_U = 2 * D_map[j, i] * D_map[j+1, i] / (D_map[j, i] + D_map[j+1, i])
                coeff_U = D_U / (h**2)
                
                # 2. 填充矩阵
                # 对角线项: (Sum of coeffs) + Sigma_removal
                diag_val = (coeff_L + coeff_R + coeff_D + coeff_U) + Sigma_rem_map[j, i]
                A[row, row] = diag_val
                
                # 非对角线项 (Neighbors)
                A[row, get_idx(j, i-1)] = -coeff_L
                A[row, get_idx(j, i+1)] = -coeff_R
                A[row, get_idx(j-1, i)] = -coeff_D
                A[row, get_idx(j+1, i)] = -coeff_U
                
        return A.tocsr() # 转换为 CSR 格式以加速求解

    print("构建快群矩阵 A1 ...")
    A1 = build_diffusion_matrix(D1_map, SigR1_map)
    print("构建热群矩阵 A2 ...")
    A2 = build_diffusion_matrix(D2_map, SigA2_map)

    # ==============================
    # 5. 源迭代求解 (Source Iteration)
    # ==============================
    print("开始迭代...")
    
    # 初始猜测
    phi1 = np.ones(N)
    phi2 = np.ones(N)
    
    # 将边界初始值设为0 (虽然这不影响求解结果，但符合物理直觉)
    for j in range(ny):
        for i in range(nx):
            if i==0 or i==nx-1 or j==0 or j==ny-1:
                idx = get_idx(j, i)
                phi1[idx] = 0.0
                phi2[idx] = 0.0

    k_eff = 1.0
    tol = 1e-6
    max_iter = 1000
    
    # 展平数组以便向量运算
    nSf1_vec = nSf1_map.flatten()
    nSf2_vec = nSf2_map.flatten()
    Ss12_vec = Ss12_map.flatten()
    
    for it in range(max_iter):
        # 1. 计算总裂变源 S_f (Total Fission Source)
        S_f = nSf1_vec * phi1 + nSf2_vec * phi2
        total_source_old = np.sum(S_f)
        
        # 2. 求解快群 (Fast Group)
        # A1 * phi1 = (1/k) * S_f
        rhs1 = S_f / k_eff
        phi1_new = spsolve(A1, rhs1)
        
        # 3. 计算散射源 (Scattering Source from Fast to Thermal)
        S_s = Ss12_vec * phi1_new
        
        # 4. 求解热群 (Thermal Group)
        # A2 * phi2 = S_s
        phi2_new = spsolve(A2, S_s)
        
        # 5. 更新 k_eff
        # k_new = k_old * (新一代总中子产额 / 旧一代总中子产额)
        # 新一代源:
        S_f_new = nSf1_vec * phi1_new + nSf2_vec * phi2_new
        total_source_new = np.sum(S_f_new)
        
        # k_eff 定义: 下一代 / 这一代
        # 这一代的有效贡献是 (S_f / k_eff)，产生 S_f_new
        k_new = total_source_new / (total_source_old / k_eff)
        
        err = abs(k_new - k_eff) / k_eff
        
        # 更新变量
        k_eff = k_new
        phi1 = phi1_new
        phi2 = phi2_new
        
        if (it+1) % 5 == 0:
            print(f"Iter {it+1:3d}: k_eff = {k_eff:.6f}, Rel.Err = {err:.2e}")
            
        if err < tol:
            print(f"\n收敛! 迭代次数: {it+1}")
            break
            
    end_time = time.time()
    print(f"计算耗时: {end_time - start_time:.2f} 秒")
    print(f"最终 k_eff = {k_eff:.5f}")

    # ==============================
    # 6. 绘图
    # ==============================
    phi1_2d = phi1.reshape((ny, nx))
    phi2_2d = phi2.reshape((ny, nx))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot Fast Flux
    im1 = axes[0].imshow(phi1_2d, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='jet', aspect='auto')
    axes[0].set_title(r'Fast Flux $\phi_1$')
    axes[0].set_xlabel('x (cm)')
    axes[0].set_ylabel('y (cm)')
    fig.colorbar(im1, ax=axes[0])
    
    # Plot Thermal Flux
    im2 = axes[1].imshow(phi2_2d, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='jet', aspect='auto')
    axes[1].set_title(r'Thermal Flux $\phi_2$')
    axes[1].set_xlabel('x (cm)')
    axes[1].set_ylabel('y (cm)')
    fig.colorbar(im2, ax=axes[1])

    # 绘制辅助线
    for ax in axes:
        # Core Box: x[-50, 50], y[15, 105]
        ax.plot([-50, 50, 50, -50, -50], [15, 15, 105, 105, 15], 'w--', lw=1.5, label='Core')
        # Interface Core 1/2: y=55
        ax.plot([-50, 50], [55, 55], 'w:', lw=1, label='Interface')
        ax.legend(loc='upper right')
        
    plt.suptitle(f'2D Reactor Flux Distribution ($k_{{eff}}={k_eff:.4f}$)', fontsize=16)
    plt.tight_layout()
    plt.show()

    plot_3d_fluxes(phi1_2d, phi2_2d, x_coords, y_coords, k_eff)

if __name__ == "__main__":
    two_diffusion()