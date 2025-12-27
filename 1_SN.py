import numpy as np
from scipy.special import roots_legendre
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def one_dimension_SN():
    # 参数设置
    a = 66.0053          # 堆尺寸 (半厚度, cm)
    Sigma_t = 0.050      # 总截面 (1/cm)
    Sigma_s = 0.030      # 散射截面 (1/cm)
    nu_Sigma_f = 0.0225  # 产额截面 (1/cm)
    c_val = (Sigma_s + nu_Sigma_f) / Sigma_t #用于验证杜华书上的值
    N_angles = 16        # Sn 阶数
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
                psi_edges[m, N_mesh] = 0.0 # 真空边界-最右侧的psi=0
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
    
    # 计算精确的边界标量通量 - 对边缘角通量积分
    phi_edge_0 = np.sum(weights * psi_edges[:, 0])      # x=0 处通量
    phi_edge_a = np.sum(weights * psi_edges[:, -1])     # x=a 处通量
    
    # 归一化基准：中心通量 phi(0)
    norm_factor = phi_edge_0
    
    # 构建全域通量数组用于插值 (包含边界点)
    # x_full = [0, center_1, center_2, ..., center_N, a]
    x_full = np.concatenate(([0], x_centers, [a]))
    phi_full = np.concatenate(([phi_edge_0], phi, [phi_edge_a]))
    
    # 归一化
    phi_normalized = phi_full / norm_factor
    
    # 创建插值函数
    f_interp = interp1d(x_full, phi_normalized, kind='cubic')
    
    # 书上的基准数据 (Table 4.5.3, c=1.05)
    ref_x_ratio = np.array([0.25, 0.50, 0.75, 1.00])
    ref_values  = np.array([0.94714400, 0.79372641, 0.55329025, 0.21419206])
    
    # 计算我们的结果
    calc_values = f_interp(ref_x_ratio * a)
    
    # 打印对比表
    print("\n" + "="*65)
    print("结果对比: 本程序计算值 vs 杜书华书 p149 表 4.5.3 (c=1.05)")
    print("="*65)
    print(f"{'x/a':^10} | {'书上值 (Ref)':^15} | {'计算值 (Calc)':^15} | {'误差 (%)':^10}")
    print("-" * 65)
    
    for i in range(4):
        ratio = ref_x_ratio[i]
        ref = ref_values[i]
        calc = calc_values[i]
        err = abs(calc - ref) / ref * 100
        print(f"{ratio:^10.2f} | {ref:^15.8f} | {calc:^15.8f} | {err:^10.4f}")
    
    print("="*65)
    print(f"有效增殖系数 k_eff: {k_eff:.6f} (理论应接近 1.0)")
    
    # 简单绘图
    plt.figure(figsize=(8,5))
    plt.plot(x_full/a, phi_normalized, 'b-', label='Calculated S16')
    plt.plot(ref_x_ratio, ref_values, 'ro', label='Reference Points')
    plt.xlabel('x/a')
    plt.ylabel(r'Normalized Flux $\phi(x)/\phi(0)$')
    plt.title('Comparison with Reference Table 4.5.3 (c=1.05)')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    one_dimension_SN()