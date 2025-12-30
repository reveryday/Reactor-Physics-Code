import numpy as np
from scipy.special import roots_legendre
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.cm as cm # 用于颜色映射

def one_dimension_SN():
    # 参数设置
    a = 66.0053          # 堆尺寸 (半厚度, cm)
    Sigma_t = 0.050      # 总截面 (1/cm)
    Sigma_s = 0.030      # 散射截面 (1/cm)
    nu_Sigma_f = 0.0225  # 产额截面 (1/cm)
    N_angles = 32        # Sn 阶数
    N_mesh = 200         # 空间网格数
    dx = a / N_mesh      # 网格步长 
    tolerance_flux = 1e-6 # 内迭代-通量收敛的判断阈值
    tolerance_k = 1e-6    # 外迭代-k的本征值收敛的判断阈值
    max_outer_iter = 2000 # 最大外迭代次数
    max_inner_iter = 1000  # 最大内迭代次数
    dx = a / N_mesh  # 网格步长

    # 初始化
    mus, weights = roots_legendre(N_angles) # 生成N_angles维高斯-勒让德求积点和权重（即有N_angles个求积节点）
    x_centers = np.linspace(dx/2, a - dx/2, N_mesh) #定义网格位置（中心点）
    phi = np.cos(np.pi * x_centers / (2 * a))  # 通量初始化：余弦分布- 200维的数组，定义在网格中心
    phi = phi / np.mean(phi)  # 归一化
    psi_edges = np.zeros((N_angles, N_mesh + 1)) # 初始化角通量 psi：16*201的二维数组，定义在网格边界上
    k_eff = 1.0

    print(f"开始计算：(网格数={N_mesh}, Sn=S{N_angles})")
    
    # 外迭代
    for outer_it in range(max_outer_iter):
        phi_old_outer = phi.copy() # 上一代通量
        source_fission = (nu_Sigma_f / k_eff) * phi #计算裂变源
        
        # 内迭代
        for inner_it in range(max_inner_iter):
            phi_old_inner = phi.copy()
            Q = Sigma_s * phi_old_inner + source_fission  # 总源项=散射项+裂变项

            # 1. Right to Left (mu < 0)扫描
            for m in range(N_angles // 2): # m=0~7，遍历前一半的角度
                mu = mus[m]  # 前一半mu索引
                psi_edges[m, N_mesh] = 0.0 # 真空边界-最右侧边界mu<0时，psi=0
                for i in range(N_mesh - 1, -1, -1): # 从最右往左遍历每个边界
                    term1 = abs(mu) / dx  # |mu|/dx
                    term2 = 0.5 * Sigma_t
                    source = 0.5 * Q[i]
                    psi_R = psi_edges[m, i+1]
                    psi_L = ((term1 - term2)*psi_R + source) / (term1 + term2)
                    if psi_L < 0: psi_L = 0.0
                    psi_edges[m, i] = psi_L # 更新左边界角通量
            
            # 2. Left to Right (mu > 0)扫描
            for m in range(N_angles // 2, N_angles):
                mu = mus[m]
                m_ref = N_angles - 1 - m
                psi_edges[m, 0] = psi_edges[m_ref, 0] # 对称边界
                for i in range(N_mesh):
                    term1 = mu / dx
                    term2 = 0.5 * Sigma_t
                    source = 0.5 * Q[i]
                    psi_L = psi_edges[m, i]
                    psi_R = ((term1 - term2)*psi_L + source) / (term1 + term2)
                    if psi_R < 0: psi_R = 0.0
                    psi_edges[m, i+1] = psi_R

            # 更新网格中心通量phi
            phi_new = np.zeros(N_mesh)
            for m in range(N_angles):
                psi_centers = 0.5 * (psi_edges[m, :-1] + psi_edges[m, 1:])
                phi_new += weights[m] * psi_centers #得到网格中心的角通量后使用gauss-legendre积分对角度积分得到phi
            phi = phi_new
            
            if np.max(np.abs(phi - phi_old_inner)/(phi_old_inner+1e-15)) < tolerance_flux:
                break
        
        # 更新 k_eff
        total_new = np.sum(phi) #总新一代通量
        total_old = np.sum(phi_old_outer) #总上一代通量
        k_new = k_eff * (total_new / total_old)
        diff_k = abs(k_new - k_eff)
        k_eff = k_new #更新k       
        
        phi = phi / np.mean(phi) # 归一化 (防溢出)
        
        if diff_k < tolerance_k:
            print(f"收敛于第 {outer_it} 步, k_eff = {k_eff:.6f}")
            break
    
    phi_0 = np.sum(weights * psi_edges[:, 0])      # x=0 处通量
    phi_a = np.sum(weights * psi_edges[:, -1])     # x=a 处通量
    

    norm_factor = phi_0  # 归一化因子 
    x_full = np.concatenate(([0], x_centers, [a])) # 坐标：x_full=0,center1,...,center200,a
    phi_full = np.concatenate(([phi_0], phi, [phi_a]))
    phi_normalized = phi_full / norm_factor # 归一化后的通量
    psi_normalized = psi_edges / norm_factor # 归一化后的角通量
    f_interp = interp1d(x_full, phi_normalized, kind='cubic') # 三次样条插值
    
    ref_x_ratio = np.array([0.25, 0.50, 0.75, 1.00]) # 书上参考点的 x/a
    ref_values  = np.array([0.94714400, 0.79372641, 0.55329025, 0.21419206])
    calc_values = f_interp(ref_x_ratio * a)

    print("结果对比:")
    print("="*65)
    print(f"{'x/a':^10} | {'Ref':^15} | {'Calc':^15} | {'误差 (%)':^10}")
    print("-" * 65)
    
    for i in range(4):
        ratio = ref_x_ratio[i]
        ref = ref_values[i]
        calc = calc_values[i]
        err = abs(calc - ref) / ref * 100
        print(f"{ratio:^10.2f} | {ref:^15.8f} | {calc:^15.8f} | {err:^10.4f}")

    # --- 图 1: 通量分布图 ---
    plt.style.use('default') 

    fig, ax1 = plt.subplots(figsize=(10, 6))
    line1, = ax1.plot(x_full/a, phi_normalized, 'b-', linewidth=2.5, label=r'Calculated Distribution (S16)')
    line2, = ax1.plot(ref_x_ratio, ref_values, 'ro', markersize=8, zorder=5, label='Reference Points')

    # 设置主坐标轴
    ax1.set_xlabel(r'$x/a$', fontsize=12)
    ax1.set_ylabel(r'$\phi(x)$', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title(rf'$\phi(x)$ ($k_{{eff}}={k_eff:.6f}$)', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 添加顶部第二坐标轴
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    new_tick_locations = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels([f"{val*a:.1f}" for val in new_tick_locations])
    ax2.set_xlabel('$x$ (cm)', fontsize=12, color='gray')
    ax2.tick_params(axis='x', colors='gray')

    # 合并图例
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.show()

    # 图2: 角通量分布图
    plt.style.use('default') 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6)) # 创建一个 1行2列 的画布

    # 左图：不同位置的角通量切片
    indices_to_plot = [0, N_mesh // 2, N_mesh] 
    labels = [r'Center ($x=0$)', r'Middle ($x \approx a/2$)', r'Right Boundary ($x=a$)']
    colors = ['blue', 'green', 'red']
    markers = ['o', '^', 's']

    for i, idx in enumerate(indices_to_plot):
        psi_at_x = psi_normalized[:, idx]
        ax1.plot(mus, psi_at_x, color=colors[i], marker=markers[i], markersize=6, 
                 linewidth=2, label=labels[i])

    ax1.axvline(0, color='black', linestyle='--', alpha=0.5) # μ=0 分界线
    ax1.set_xlabel(r'Direction Cosine $\mu$', fontsize=12)
    ax1.set_ylabel(r'Normalized Angular Flux $\psi(x, \mu)$', fontsize=12)
    ax1.set_title(r'(a) Angular Flux Anisotropy at Different Locations', fontsize=14)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=10)
    
    # 添加物理标注
    ax1.text(-0.8, psi_normalized[:, 0].max()*0.1, 'Incoming\n(Left)', ha='center', color='gray')
    ax1.text(0.8, psi_normalized[:, 0].max()*0.1, 'Outgoing\n(Right)', ha='center', color='gray')

    # 右图：相空间热力图 (Phase Space Heatmap)
    x_edges_grid = np.linspace(0, a, N_mesh + 1)
    X, Y = np.meshgrid(x_edges_grid/a, mus)
    contour = ax2.contourf(X, Y, psi_normalized, 100, cmap='jet')
    # 添加颜色条
    cbar = plt.colorbar(contour, ax=ax2)
    cbar.set_label(r'$\phi$', rotation=270, labelpad=15)

    ax2.set_xlabel(r'$x/a$', fontsize=12)
    ax2.set_ylabel(r'$\mu$', fontsize=12)
    ax2.set_title(r'$\phi(x, \mu)$', fontsize=14)
    
    # 画出 μ=0 的分界线
    ax2.axhline(0, color='white', linestyle='--', linewidth=1.5)
    ax2.text(0.05, 0.5, 'Forward ($\mu>0$)', color='white', fontweight='bold', fontsize=10)
    ax2.text(0.05, -0.5, 'Backward ($\mu<0$)', color='white', fontweight='bold', fontsize=10)

    # 标注真空边界特征
    ax2.annotate('Vacuum Boundary\n(Zero Flux)', xy=(0.95, -0.5), xytext=(0.6, -0.8),
                 color='white', arrowprops=dict(facecolor='white', arrowstyle='->'))

    plt.tight_layout()
    plt.show()

    # 图3: 3D 角通量分布图
    from mpl_toolkits.mplot3d import Axes3D  # 必须引入这个模块
    
    plt.style.use('default')
    x_grid = np.linspace(0, a, N_mesh + 1) / a  # 归一化位置
    X, Y = np.meshgrid(x_grid, mus)
    Z = psi_normalized  # 高度 Z 就是角通量值
    
    # 2. 创建 3D 画布
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='plasma', 
                           edgecolor='none',  # 去掉网格线让颜色更平滑
                           antialiased=True,  # 抗锯齿
                           alpha=0.9)         # 稍微透明一点点
    ax.view_init(elev=30, azim=-135)
    
    # 5. 坐标轴标签
    ax.set_xlabel(r'Position $x/a$', fontsize=12, labelpad=10)
    ax.set_ylabel(r'Direction $\mu$', fontsize=12, labelpad=10)
    ax.set_zlabel(r'Angular Flux $\psi$', fontsize=12, labelpad=10)
    ax.set_title(r'3D Visualization of Angular Flux $\phi(x, \mu)$', fontsize=15)
    
    # 6. 添加颜色条
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, pad=0.1)
    cbar.set_label(r'Flux Intensity', rotation=270, labelpad=15)

    ax.plot([0, 1], [0, 0], [0, 0], 'w--', linewidth=1, zorder=10) # 地面上的中线
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    one_dimension_SN()