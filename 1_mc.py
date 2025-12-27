import numpy as np
import matplotlib.pyplot as plt

class SlabReactorMC:
    def __init__(self, num_neutrons=5000, num_cycles=100, num_inactive=30, num_bins=100):
        """
        :param num_neutrons: 每代的模拟中子数 (粒子数)
        :param num_cycles: 总迭代代数
        :param num_inactive: 非活跃代数 (用于源收敛，不计入统计)
        :param num_bins: 空间网格数 (用于统计通量)
        """
        self.num_neutrons = num_neutrons
        self.num_cycles = num_cycles
        self.num_inactive = num_inactive
        self.num_bins = num_bins
        
        # 几何定义 (单位: cm)
        self.width = 100.0
        self.interface = 50.0
        self.dx = self.width / self.num_bins
        
        # 材料参数定义
        # Region 1: 0 < x < 50
        # Sig_a = 0.12, Sig_s = 0.05, nu_Sig_f = 0.15
        # Sig_t = Sig_a + Sig_s = 0.17
        self.mat1 = {
            'Sig_t': 0.17,
            'Sig_s': 0.05,
            'Sig_a': 0.12,
            'nu_Sig_f': 0.15,
            'P_scatter': 0.05 / 0.17  # 散射概率 Sig_s / Sig_t
        }
        
        # Region 2: 50 < x < 100
        # Sig_a = 0.10, Sig_s = 0.05, nu_Sig_f = 0.12
        # Sig_t = Sig_a + Sig_s = 0.15
        self.mat2 = {
            'Sig_t': 0.15,
            'Sig_s': 0.05,
            'Sig_a': 0.10,
            'nu_Sig_f': 0.12,
            'P_scatter': 0.05 / 0.15
        }

        # 结果容器
        self.k_eff_history = []
        self.flux_tally = np.zeros(self.num_bins)
        self.flux_squared_tally = np.zeros(self.num_bins) # 用于计算方差 (可选)
        
    def get_material(self, x):
        """根据位置返回材料属性"""
        if 0.0 <= x < self.interface:
            return self.mat1
        elif self.interface <= x <= self.width:
            return self.mat2
        else:
            return None # 真空/泄漏

    def run(self):
        print(f"开始蒙特卡罗模拟...")
        print(f"粒子数/代: {self.num_neutrons}, 总代数: {self.num_cycles}")
        
        # 1. 初始源分布：在 0-100cm 均匀分布
        source_sites = np.random.uniform(0, self.width, self.num_neutrons)
        
        # 主循环：代（Generations）
        for cycle in range(self.num_cycles):
            new_fission_sites = [] # 下一代的中子产生位置
            cycle_flux = np.zeros(self.num_bins)
            
            # 循环：当前代的所有中子
            for i in range(self.num_neutrons):
                x = source_sites[i]
                
                # 初始方向：各向同性 (Isotropic)
                # 在一维平板中，mu = cos(theta) 在 [-1, 1] 均匀分布
                mu = np.random.uniform(-1.0, 1.0)
                
                alive = True
                while alive:
                    mat = self.get_material(x)
                    if mat is None: # 粒子已经在几何体外 (这通常不应在循环开始发生)
                        alive = False
                        break
                        
                    Sig_t = mat['Sig_t']
                    
                    # 采样飞行距离 (Distance to Collision)
                    # d = -ln(xi) / Sig_t
                    d_collision = -np.log(np.random.random()) / Sig_t
                    
                    # 计算到边界的距离
                    if mu > 0:
                        dist_to_boundary = (self.interface - x) if x < self.interface else (self.width - x)
                        next_region_x = x + dist_to_boundary + 1e-10 # 稍微推过边界
                    elif mu < 0:
                        dist_to_boundary = (x - self.interface) if x > self.interface else x
                        next_region_x = x - dist_to_boundary - 1e-10
                    else:
                        # mu = 0 (极其罕见)，在平板中无限运动，强制重采样
                        mu = np.random.uniform(-1.0, 1.0)
                        continue
                        
                    # 几何距离 = 物理距离 / |mu|
                    # 如果 mu 接近 0，步长会很大，这是物理正确的
                    geo_dist_to_boundary = dist_to_boundary / abs(mu)

                    # 判断: 碰撞 还是 穿越边界/泄漏?
                    if d_collision < geo_dist_to_boundary:
                        # --- 发生碰撞 ---
                        dist_moved = d_collision
                        dx_moved = dist_moved * mu
                        
                        # 通量统计 (Track Length Estimator): flux += path_length / volume
                        # 这里是一维，Volume 等效于 dx。实际上我们先累加 path length，最后归一化
                        # 粒子移动过程可能跨越多个网格，为简化代码，我们将通量加在碰撞点所在的网格
                        # (对于细网格，更精确的做法是计算穿过每个网格的长度，但在本作业级别，取中点或终点即可)
                        end_x = x + dx_moved
                        bin_idx = int(end_x / self.dx)
                        if 0 <= bin_idx < self.num_bins:
                            cycle_flux[bin_idx] += d_collision / abs(mu) # 通量 = 径迹长度 (1/Sigma_t 贡献)
                            # 注意: d_collision 是光学路径对应的物理长度? 
                            # 不，d_collision就是物理长度。Flux = Sum(path_length).
                            # wait, standard MC: d sampled is physical distance.
                            cycle_flux[bin_idx] += d_collision
                        
                        # 移动粒子
                        x = end_x
                        
                        # 决定反应类型：散射 还是 吸收
                        if np.random.random() < mat['P_scatter']:
                            # 散射：各向同性重新采样方向，继续飞行
                            mu = np.random.uniform(-1.0, 1.0)
                        else:
                            # 吸收：历史终止
                            alive = False
                            # 产生下一代中子 (裂变)
                            # 期望产生的裂变中子数 = nu * (Sig_f / Sig_a) 
                            # 注意：题目给出的是 nu*Sig_f 和 Sig_a
                            # 产生概率权重 weight = nu_Sig_f / Sig_a
                            expected_nu = mat['nu_Sig_f'] / mat['Sig_a']
                            
                            # 使用俄罗斯轮盘赌或直接统计期望来存入下一代
                            # 为了保持粒子数稳定，我们记录期望值，稍后重采样
                            # 或者简单的整数化处理: int(expected_nu + random)
                            num_produced = int(expected_nu + np.random.random())
                            for _ in range(num_produced):
                                new_fission_sites.append(x)
                    
                    else:
                        # --- 穿越边界 或 泄漏 ---
                        # 移动到边界
                        dist_moved = geo_dist_to_boundary
                        end_x = next_region_x 
                        
                        # 简单的通量统计（加上这段路径）
                        # 同样简化处理，加在出发点所在的网格（或者更精确地处理跨网格）
                        # 这里为了代码简洁，加在当前位置。误差随网格加密减小。
                        bin_idx = int(x / self.dx) 
                        if 0 <= bin_idx < self.num_bins:
                            cycle_flux[bin_idx] += dist_moved
                        
                        x = end_x
                        
                        # 检查是否泄漏
                        if x < 0.0 or x > self.width:
                            alive = False # 泄漏死亡

            # --- 代结束处理 ---
            
            # 1. 计算 k_eff
            # k = (产生的新中子数) / (上一代源中子数)
            # 但为了防止粒子数爆炸或消失，我们通常强制归一化源
            k_cycle = len(new_fission_sites) / self.num_neutrons
            
            # 2. 记录状态
            if cycle >= self.num_inactive:
                self.k_eff_history.append(k_cycle)
                self.flux_tally += cycle_flux # 累加通量
            
            # 3. 准备下一代源 (Shannon Entropy / Resampling)
            if len(new_fission_sites) == 0:
                print("警告: 中子全部消失，反应堆处于极深次临界。")
                break
                
            # 从裂变库中随机抽取 num_neutrons 个位置作为下一代源
            # 这保证了粒子数恒定
            source_sites = np.random.choice(new_fission_sites, self.num_neutrons)
            
            # 打印进度
            if (cycle + 1) % 10 == 0:
                print(f"Cycle {cycle+1}/{self.num_cycles}, k_eff = {k_cycle:.5f}")

        self.post_process()

    def post_process(self):
        """数据处理与绘图"""
        # 计算 K_eff 统计量
        avg_k = np.mean(self.k_eff_history)
        std_k = np.std(self.k_eff_history)
        
        print("-" * 30)
        print(f"计算完成 (活跃代数: {self.num_cycles - self.num_inactive})")
        print(f"平均 k_eff = {avg_k:.5f} +/- {std_k:.5f}")
        print("-" * 30)
        
        # 归一化通量
        # 此时 flux_tally 是所有活跃代所有粒子的路径长度总和
        # Flux ~ sum(track_length) / (Volume * N_active_cycles * N_particles)
        # 这里是一维，Volume = dx
        # 我们将其归一化，使得最大值为1，方便观察形状
        
        x_axis = np.linspace(0, self.width, self.num_bins)
        norm_factor = np.mean(self.flux_tally) # 归一化到平均值为1
        # 或者归一化到体积积分
        final_flux = self.flux_tally / (self.dx * (self.num_cycles - self.num_inactive) * self.num_neutrons)
        
        # 绘图
        plt.figure(figsize=(10, 6))
        
        # 绘制通量
        plt.plot(x_axis, final_flux, 'b-', label='Neutron Flux', linewidth=2)
        
        # 绘制区域分界线
        plt.axvline(x=50, color='r', linestyle='--', label='Material Interface (50cm)')
        
        plt.xlabel('Position x (cm)')
        plt.ylabel('Flux (Arbitrary Units)')
        plt.title(f'Flux Distribution (Slab Reactor)\n k_eff = {avg_k:.5f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # 绘制 K_eff 收敛图
        plt.figure(figsize=(8, 4))
        plt.plot(range(self.num_inactive, self.num_cycles), self.k_eff_history, marker='o', markersize=3)
        plt.axhline(y=avg_k, color='r', linestyle='--', label='Mean k_eff')
        plt.xlabel('Active Cycle')
        plt.ylabel('k_eff')
        plt.title('k_eff Convergence')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # 建议参数：
    # 粒子数越多，方差越小，曲线越平滑
    # 代数越多，k_eff 越准确
    sim = SlabReactorMC(num_neutrons=10000, num_cycles=150, num_inactive=20, num_bins=50)
    sim.run()