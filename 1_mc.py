import numpy as np
import matplotlib.pyplot as plt

N_PARTICLES = 100000   # 每代粒子数
N_ACTIVE = 500        # 活跃代数
N_INACTIVE = 20       # 非活跃代数
N_BINS = 100          # 空间网格数

WIDTH = 100.0         # 反应堆总宽度 (cm)
INTERFACE = 50.0      # 材料交界面位置 (cm)
DX = WIDTH / N_BINS   # 网格宽度

# 根据坐标x获取截面信息
def get_crosssection(x):
    # Region 1: 0 < x < 50
    if 0 <= x < INTERFACE:
        Sa = 0.12
        Ss = 0.05
        nSf = 0.15
    # Region 2: 50 <= x <= 100
    else: 
        Sa = 0.10
        Ss = 0.05
        nSf = 0.12
        
    St = Sa + Ss
    return Sa, Ss, nSf, St

# d是沿着飞行方向的飞行距离，dx_proj是投影到x距离
def tally_track_length(x_start, x_end, d, dx_proj, flux_tally):  

    limit_min, limit_max = 0.0, WIDTH  # 定义有效边界
    
    # 截断：限制了在[0, WIDTH]范围内
    x0 = max(min(x_start, limit_max), limit_min) # 经过截断之后的初始坐标x_0
    x1 = max(min(x_end, limit_max), limit_min)  # 经过截断之后的结束坐标x_1
    dist_proj = abs(x1 - x0) # 截断之后中子在x轴上的飞行距离
    if dist_proj < 1e-12:
        return
    
    if abs(dx_proj) > 1e-12:
        fraction = dist_proj / abs(dx_proj)  # 有效路径长度占总路径长度的比例
        d_eff = d * fraction # 截断后的有效路径长度
    else:
        d_eff = d  # 垂直飞行

    mid_point = (x0 + x1) / 2.0 # 中子路径的中点位置
    bin_idx = int(mid_point / DX) # 找到对应的网格索引
    
    if 0 <= bin_idx < N_BINS: # 确保索引在有效范围内
        flux_tally[bin_idx] += d_eff 

def transport_one_generation(source_bank, is_active, flux_tally):

    next_gen_bank = [] #下一代源
    
    # 模拟每一个中子
    for x in source_bank:
        mu = np.random.uniform(-1, 1) # 初始化方向 -各向同性分布
        alive = True    
            
        while alive:          
            Sa, Ss, nSf, St = get_crosssection(x)  # 获取当前位置材料属性
            d = -np.log(np.random.random()) / St # 2. 采样飞行距离 d
            
            # 3. 几何移动
            dx_proj = d * mu
            x_new = x + dx_proj
            
            # --- 通量统计 ---
            if is_active:
                tally_track_length(x, x_new, d, dx_proj, flux_tally)
            
            # 4. 边界检查 (Leakage)
            if x_new < 0 or x_new > WIDTH:
                alive = False
                continue
            
            # 更新位置
            x = x_new
            
            # 更新材料 (位置改变导致材料可能改变)
            Sa, Ss, nSf, St = get_crosssection(x)
            
            # 5. 碰撞反应判定
            if np.random.random() < (Ss / St):
                # -> 散射
                mu = np.random.uniform(-1, 1)
            else:
                # -> 吸收 -> 裂变
                alive = False
                
                # 计算期望裂变中子数
                nu_expected = nSf / Sa
                
                # 随机整数化处理
                n_prod = int(nu_expected)
                if np.random.random() < (nu_expected - n_prod):
                    n_prod += 1
                
                # 存入下一代银行
                for _ in range(n_prod):
                    next_gen_bank.append(x)
                    
    return next_gen_bank

def resample_source(next_gen_bank, target_size):
    """源重采样"""
    if len(next_gen_bank) == 0:
        return [] 
        
    if len(next_gen_bank) > target_size:
        return np.random.choice(next_gen_bank, target_size, replace=False)
    else:
        return np.random.choice(next_gen_bank, target_size, replace=True)

def run_monte_carlo():
    """主执行函数"""
    # 初始化
    flux_tally = np.zeros(N_BINS)
    k_history = []
    
    # 初始源
    current_source = np.random.uniform(0, WIDTH, N_PARTICLES)
    total_gens = N_ACTIVE + N_INACTIVE
    
    print(f"--- MC Simulation Start: {N_PARTICLES} particles, {total_gens} generations ---")
    
    # 代循环
    for gen in range(total_gens):
        is_active = (gen >= N_INACTIVE)
        
        # --- 核心输运 (参数里不再需要 config) ---
        next_source = transport_one_generation(current_source, is_active, flux_tally)
        
        # --- 计算 k_eff ---
        if len(current_source) > 0:
            k_eff = len(next_source) / len(current_source)
        else:
            k_eff = 0.0
            
        if is_active:
            k_history.append(k_eff)
        
        # 打印进度
        if (gen + 1) % 10 == 0:
            status = "Active" if is_active else "Inactive"
            print(f"Gen {gen+1:3d} [{status}]: k = {k_eff:.5f}")
        
        # --- 源重采样 ---
        if len(next_source) == 0:
            print("Error: Reactor went subcritical.")
            break
        current_source = resample_source(next_source, N_PARTICLES)

    # 结果整理
    avg_k = np.mean(k_history)
    std_k = np.std(k_history)
    
    # 通量归一化
    norm_factor = DX * N_ACTIVE * N_PARTICLES
    flux = flux_tally / norm_factor
    
    return avg_k, std_k, flux

def plot_results(avg_k, std_k, flux):
    """
    绘制结果
    直接使用全局变量 N_BINS, WIDTH, DX, INTERFACE
    """
    # 归一化通量
    flux_norm = flux / np.max(flux)
    
    # 生成 x 轴
    x_axis = np.linspace(DX/2, WIDTH - DX/2, N_BINS)
    
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(x_axis, flux_norm, 'b.-', label=r'Normalized Flux $\phi(x)$', linewidth=1.5, markersize=4)
    plt.axvline(x=INTERFACE, color='r', linestyle='--', alpha=0.8, label='Interface (50cm)')
    
    plt.xlabel('x (cm)', fontsize=12)
    plt.ylabel(r'$\phi(x)$', fontsize=12)
    plt.title(rf'MC Flux Distribution' + '\n' + 
              rf'$k_{{eff}} = {avg_k:.4f} \pm {std_k:.4f}$', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(0, WIDTH)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 运行计算
    k_val, k_std, flux_dist = run_monte_carlo()
    
    # 输出结果
    print("="*30)
    print(f"Final Result: k_eff = {k_val:.5f} +/- {k_std:.5f}")
    print("="*30)
    
    # 绘图
    plot_results(k_val, k_std, flux_dist)