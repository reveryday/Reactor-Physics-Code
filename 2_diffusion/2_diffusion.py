import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import time

"""所求的通量phi[j,i]是定义在网格节点处的，将其展平后就是一个N维向量"""

# [Group 1, Group 2] -字典形式
mat_c1 = {'D': [1.267, 0.354], 'Sigma_a': [0.0121, 0.121], 'nu_Sigma_f': [0.0085, 0.1851], 'Sigma_s12': 0.0241}
mat_c2 = {'D': [1.280, 0.400], 'Sigma_a': [0.010, 0.100],  'nu_Sigma_f': [0.006, 0.150],  'Sigma_s12': 0.016}
mat_re = {'D': [1.130, 0.166], 'Sigma_a': [0.0004, 0.020], 'nu_Sigma_f': [0.0, 0.0],      'Sigma_s12': 0.0493}  # 反射层
x_min, x_max = -65.0, 65.0
y_min, y_max = 0.0, 120.0   
h = 0.125 # 网格步长   
nx = int(round((x_max - x_min) / h)) + 1 # 避免浮点数误差
ny = int(round((y_max - y_min) / h)) + 1
N = nx * ny # 总节点数

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

def get_idx(j, i, nx): # 第j行第i个
        return j * nx + i # 得到展开成向量后的索引

def two_diffusion():
    start_time = time.time()

    x_coords = np.linspace(x_min, x_max, nx) # 生成x坐标：从-65到 65共131个点
    y_coords = np.linspace(y_min, y_max, ny) # 生成y坐标：从0到120共121个点
    
    # 构造了7个二维矩阵-材料矩阵
    D1_map = np.zeros((ny, nx)) # 快中子扩散系数矩阵 [121, 131]
    D2_map = np.zeros((ny, nx)) # 热中子扩散系数矩阵
    SigR1_map = np.zeros((ny, nx)) # 快中子移出截面矩阵
    SigA2_map = np.zeros((ny, nx)) # 热中子吸收截面矩阵
    nSf1_map = np.zeros((ny, nx)) # 快中子产额截面矩阵
    nSf2_map = np.zeros((ny, nx)) # 热中子产额截面矩阵
    Ss12_map = np.zeros((ny, nx)) # 快到热散射截面矩阵
    
    # 得到了每个网格节点处的材料矩阵，每一种材料矩阵都是[121, 131]
    for j in range(ny):
        y = y_coords[j] # 第j行的y坐标
        for i in range(nx):
            x = x_coords[i] # 第i列的x坐标

            in_x_core = -50.0 < x < 50.0
            if in_x_core and (15.0 < y < 55.0):
                m = mat_c1 # 字典格式，键为材料编号，值为材料对象
            elif in_x_core and (55.0 <= y < 105.0):
                m = mat_c2
            else:
                m = mat_re
            
            D1_map[j, i] = m['D'][0] # 字典的第一个元素代表快群群常数
            D2_map[j, i] = m['D'][1]
            SigR1_map[j, i] = m['Sigma_a'][0] + m['Sigma_s12'] 
            SigA2_map[j, i] = m['Sigma_a'][1] 
            nSf1_map[j, i] = m['nu_Sigma_f'][0]
            nSf2_map[j, i] = m['nu_Sigma_f'][1]
            Ss12_map[j, i] = m['Sigma_s12'] 

    def build_diffusion_matrix(D_map, Sigma_rem_map):
        A = lil_matrix((N, N)) # 构建稀疏矩阵[N, N] --> 加快计算速度   

        # 构建系数矩阵A-矩阵A只有对角线跟对角线上下左右有值
        for j in range(ny): # 遍历每一个网格节点
            for i in range(nx):
                row = get_idx(j, i, nx)
                
                if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                    A[row, row] = 1.0
                    continue
                
                # Left (i-1)
                D_L = 2 * D_map[j, i] * D_map[j, i-1] / (D_map[j, i] + D_map[j, i-1])
                coeff_L = D_L / (h**2) # 泄露项带来的系数
                
                # Right (i+1)
                D_R = 2 * D_map[j, i] * D_map[j, i+1] / (D_map[j, i] + D_map[j, i+1])
                coeff_R = D_R / (h**2)
                
                # Down (j-1)
                D_D = 2 * D_map[j, i] * D_map[j-1, i] / (D_map[j, i] + D_map[j-1, i])
                coeff_D = D_D / (h**2)
                
                # Up (j+1)
                D_U = 2 * D_map[j, i] * D_map[j+1, i] / (D_map[j, i] + D_map[j+1, i])
                coeff_U = D_U / (h**2)
                
                diag_val = (coeff_L + coeff_R + coeff_D + coeff_U) + Sigma_rem_map[j, i]
                A[row, row] = diag_val
                
                # 非对角线项
                A[row, get_idx(j, i-1, nx)] = -coeff_L
                A[row, get_idx(j, i+1, nx)] = -coeff_R
                A[row, get_idx(j-1, i, nx)] = -coeff_D
                A[row, get_idx(j+1, i, nx)] = -coeff_U
                
        return A.tocsr()

    A1 = build_diffusion_matrix(D1_map, SigR1_map)
    A2 = build_diffusion_matrix(D2_map, SigA2_map)

    # 初始化
    phi1 = np.ones(N)
    phi2 = np.ones(N)
    k_eff = 1.0
    tol = 1e-6
    max_iter = 1000  
    # 将边界初始值设为0
    for j in range(ny):
        for i in range(nx):
            if i==0 or i==nx-1 or j==0 or j==ny-1:
                idx = get_idx(j, i, nx)
                phi1[idx] = 0.0
                phi2[idx] = 0.0
   
    nSf1_vec = nSf1_map.flatten() # 展平数组以便向量运算
    nSf2_vec = nSf2_map.flatten()
    Ss12_vec = Ss12_map.flatten()
    # 记录 k_eff 变化用于绘图
    k_history = [k_eff]
    
    # 源迭代
    for it in range(max_iter):
        
        S_f = nSf1_vec * phi1 + nSf2_vec * phi2 # 计算总裂变源 S_f
        total_source_old = np.sum(S_f) # 上一代总源
        
        # 求解快群
        rhs1 = S_f / k_eff   # A1 * phi1 = (1/k) * S_f
        """线性方程组右边的b是该节点对应的裂变源，系数矩阵是"""
        phi1_new = spsolve(A1, rhs1)  # 求解线性方程组: A1 * phi1 = rhs1
        S_s = Ss12_vec * phi1_new  # 计算散射源
        phi2_new = spsolve(A2, S_s)
        S_f_new = nSf1_vec * phi1_new + nSf2_vec * phi2_new
        total_source_new = np.sum(S_f_new)
        k_new = total_source_new / (total_source_old / k_eff)        
        err = abs(k_new - k_eff) / k_eff
        
        # 更新变量
        k_eff = k_new
        k_history.append(k_eff)
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

    # 绘制 k_eff 随迭代次数变化曲线
    try:
        import matplotlib.pyplot as _plt
        iters = list(range(len(k_history)))
        _plt.figure(figsize=(6,4))
        _plt.plot(iters, k_history, marker='o')
        _plt.xlabel('Iteration')
        _plt.ylabel(r'$k_{eff}$')
        _plt.title('$k_{eff}$ vs Iteration')
        _plt.grid(True)
        _plt.tight_layout()
        #_plt.show()
    except Exception:
        pass

    def plot_results():
        # 绘图
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
        #plt.show()
    
        #plot_3d_fluxes(phi1_2d, phi2_2d, x_coords, y_coords, k_eff)

    #plot_results()

if __name__ == "__main__":
    two_diffusion()