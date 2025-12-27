import numpy as np
from scipy.special import roots_legendre
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def solve_and_compare():
    # ==============================
    # 1. 物理参数定义
    # ==============================
    a = 66.0053          # 堆尺寸 (半厚度, cm)
    Sigma_t = 0.050      # 总截面
    Sigma_s = 0.030      # 散射截面
    nu_Sigma_f = 0.0225  # 产额截面
    
    # 验证 c 值
    c_val = (Sigma_s + nu_Sigma_f) / Sigma_t
    print(f"当前物理参数对应的 c 值: {c_val:.2f} (对应表4.5.3中的 c=1.05 行)")

    # ==============================
    # 2. 数值计算参数
    # ==============================
    N_angles = 16        # S16 (角度离散越高越准)
    N_mesh = 200         # 空间网格数 (越密越准)
    dx = a / N_mesh
    
    tolerance_flux = 1e-6
    tolerance_k = 1e-6
    max_outer_iter = 2000
    
    # ==============================
    # 3. 初始化
    # ==============================
    mus, weights = roots_legendre(N_angles)
    x_centers = np.linspace(dx/2, a - dx/2, N_mesh)
    
    # 初始猜测 (余弦分布)
    phi = np.cos(np.pi * x_centers / (2 * a))
    phi = phi / np.mean(phi)
    
    psi_edges = np.zeros((N_angles, N_mesh + 1))
    k_eff = 1.0

    print(f"开始计算... (网格数={N_mesh}, Sn=S{N_angles})")
    
    # ==============================
    # 4. 迭代求解 (标准 SN 流程)
    # ==============================
    for outer_it in range(max_outer_iter):
        phi_old_outer = phi.copy()
        
        # 裂变源 (外迭代固定)
        source_fission = (nu_Sigma_f / k_eff) * phi 
        
        # 内迭代
        for inner_it in range(100):
            phi_old_inner = phi.copy()
            Q = Sigma_s * phi_old_inner + source_fission
            
            # --- Sweep ---
            # 1. Right to Left (mu < 0)
            for m in range(N_angles // 2):
                mu = mus[m] 
                psi_edges[m, N_mesh] = 0.0 # 真空边界
                for i in range(N_mesh - 1, -1, -1):
                    term1 = abs(mu) / dx
                    term2 = 0.5 * Sigma_t
                    source = 0.5 * Q[i]
                    psi_R = psi_edges[m, i+1]
                    psi_L = ((term1 - term2)*psi_R + source) / (term1 + term2)
                    if psi_L < 0: psi_L = 0.0
                    psi_edges[m, i] = psi_L
            
            # 2. Left to Right (mu > 0)
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

            # 更新 Flux (网格中心)
            phi_new = np.zeros(N_mesh)
            for m in range(N_angles):
                psi_centers = 0.5 * (psi_edges[m, :-1] + psi_edges[m, 1:])
                phi_new += weights[m] * psi_centers
            phi = phi_new
            
            if np.max(np.abs(phi - phi_old_inner)/(phi_old_inner+1e-15)) < tolerance_flux:
                break
        
        # 更新 k_eff
        total_new = np.sum(phi)
        total_old = np.sum(phi_old_outer)
        k_new = k_eff * (total_new / total_old)
        diff_k = abs(k_new - k_eff)
        k_eff = k_new
        
        # 归一化 (防溢出)
        phi = phi / np.mean(phi)
        
        if diff_k < tolerance_k:
            print(f"收敛于第 {outer_it} 步, k_eff = {k_eff:.6f}")
            break

    # ==============================
    # 5. 数据处理与对比 (核心部分)
    # ==============================
    
    # 计算精确的边界标量通量 (通过对边缘角通量积分)
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
    solve_and_compare()