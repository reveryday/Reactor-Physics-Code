import numpy as np
import os
from scipy.special import roots_legendre
import matplotlib.pyplot as plt

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

mus, weights = roots_legendre(N_angles) # 生成N_angles维高斯-勒让德求积点和权重（即有N_angles个求积节点）
x_centers = np.linspace(dx/2, a - dx/2, N_mesh) #定义网格位置（中心点）

# 内迭代：裂变源固定-更新phi
def inner_iteration(psi_edges,k_eff):
    for inner_it in range(max_inner_iter):

        phi_old_inner = phi.copy() #上一代通量       
        scattering_source=Sigma_s * phi_old_inner # 散射源项
        source_fission = (nu_Sigma_f / k_eff) * phi # 计算裂变源项，其在内迭代中保持不变

        Q = scattering_source + source_fission # 总源项=散射项+裂变项
        
        # 1. 负方向扫描 (mu < 0): 右 -> 左
        for m in range(N_angles // 2):

            mu = mus[m] # m=1~6（前一半索引）         
            psi_edges[m, N_mesh] = 0.0 # 右边界条件: 真空 (psi = 0)
            
            for i in range(N_mesh - 1, -1, -1): #最右边的网格对应N_mesh - 1
                # 菱形差分公式
                term1 = abs(mu) / dx
                term2 = 0.5 * Sigma_t
                # 输运方程右端项是 0.5 * Q (各向同性源均匀分配到所有角度)
                source_term = 0.5 * Q[i]
                
                psi_R = psi_edges[m, i+1] # 入射
                
                # psi_L = ( (term1 - term2)*psi_R + source_term ) / (term1 + term2)
                numerator = (term1 - term2) * psi_R + source_term
                denominator = term1 + term2
                psi_L = numerator / denominator
                
                # 负通量修正
                if psi_L < 0: psi_L = 0.0
                
                psi_edges[m, i] = psi_L
        
        # 2. 正方向扫描 (mu > 0): 左 -> 右
        for m in range(N_angles // 2, N_angles):
            mu = mus[m]
            # 左边界: 反射 (psi(mu) = psi(-mu))
            m_reflect = N_angles - 1 - m
            psi_edges[m, 0] = psi_edges[m_reflect, 0]
            
            for i in range(N_mesh):
                term1 = mu / dx
                term2 = 0.5 * Sigma_t
                source_term = 0.5 * Q[i]
                
                psi_L = psi_edges[m, i] # 入射
                
                numerator = (term1 - term2) * psi_L + source_term
                denominator = term1 + term2
                psi_R = numerator / denominator
                
                if psi_R < 0: psi_R = 0.0
                    
                psi_edges[m, i+1] = psi_R

        # 更新标量通量
        phi_new = np.zeros(N_mesh)
        for m in range(N_angles):
            # 菱形差分假设：中心通量是左右平均
            psi_centers = 0.5 * (psi_edges[m, :-1] + psi_edges[m, 1:])
            phi_new += weights[m] * psi_centers
        
        phi = phi_new
        
        # 内迭代判敛
        denom = np.maximum(np.abs(phi_old_inner), 1e-12)
        max_err = np.max(np.abs(phi - phi_old_inner) / denom)
        if max_err < tolerance_flux:
            break

    return phi

# 外迭代：更新 k_eff
def outer_iteration(): 

    # 初始化
    phi = np.cos(np.pi * x_centers / (2 * a)) # 标量通量初始化：余弦分布
    phi = phi / np.mean(phi) # 归一化初始通量   
    psi_edges = np.zeros((N_angles, N_mesh + 1)) # 初始化角通量 psi：16*201的二维数组，定义在网格边界上
    k_eff = 1.0 # 初始 k_eff
    print(f"开始计算: S{N_angles}, 网格数={N_mesh}, 尺寸 a={a} cm, Sn= {N_angles}")

    # 外迭代
    for outer_it in range(max_outer_iter):
        phi_old_outer = phi.copy() # 保存上一代通量用于计算 k（父代phi）        

        inner_iteration(psi_edges,k_eff) # 内迭代更新标量通量
        
        # 更新keff
        total_production_new = np.sum(phi)      # 当前这一代的总通量
        total_production_old = np.sum(phi_old_outer) # 上一代的总通量
        
        k_new = k_eff * (total_production_new / total_production_old)
        diff_k = abs(k_new - k_eff)
        k_eff = k_new
        
        # 每次外迭代后必须归一化，防止数值溢出
        phi_mean = float(np.mean(phi))
        if not np.isfinite(phi_mean) or phi_mean == 0.0:
            raise FloatingPointError("Invalid flux mean encountered during normalization")
        phi = phi / phi_mean
        
        if outer_it % 5 == 0 or outer_it < 5:
            print(f"Iter {outer_it:3d}: k_eff = {k_eff:.6f}, k_diff = {diff_k:.2e}")
            
        # 外迭代判敛
        if diff_k < tolerance_k:
            print("-" * 60)
            print(f"收敛于第 {outer_it} 步")
            break

if __name__ == "__main__":
    outer_iteration()
