import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # 进度条库，没有的话可以把相关代码删掉

class Segment:
    """定义特征线上的线段：属于哪个网格，长度是多少"""
    def __init__(self, region_idx, length):
        self.region_idx = region_idx # 网格索引 (ix, iy)
        self.length = length         # 线段长度 (cm)

class Track:
    """定义一条特征线"""
    def __init__(self, start_pt, angle_idx, spacing):
        self.segments = []      # 包含的线段列表
        self.angle_idx = angle_idx 
        self.spacing = spacing  # 线间距 (cm)
        self.flux_in = 0.0      # 入射角通量 (真空边界默认为0)

class MOC_Solver_2D:
    def __init__(self, side_length, n_mesh, n_angles):
        # --- 1. 几何与网格定义 ---
        self.L = side_length          # 反应堆边长 (cm)
        self.N = n_mesh               # 网格划分 N x N
        self.dx = self.L / self.N     # 网格尺寸
        self.n_angles = n_angles      # 方位角数量 (0到90度划分数)
        
        # 物理参数 (默认为均匀介质，类似第1题的参数)
        # 这里把 2D 数组初始化，支持非均匀介质 (Checkerboard)
        self.sigma_t = np.ones((self.N, self.N)) * 0.5   # 总截面
        self.sigma_s = np.ones((self.N, self.N)) * 0.25  # 散射截面
        self.nu_sigma_f = np.ones((self.N, self.N)) * 0.255 # 产额截面 (设置得稍大以保证临界)
        
        # 结果数组
        self.flux = np.ones((self.N, self.N)) # 标量通量
        self.k_eff = 1.0
        
        # MOC 射线参数
        self.tracks = []     # 所有的射线
        self.angles = []     # 角度值
        self.weights = []    # 角度权重
        
        # 初始化
        self._init_angles()
        self._perform_ray_tracing()

    def _init_angles(self):
        """初始化求积组 (Tabuchi-Yamamoto 或简单的均匀分布)"""
        # 简单起见，我们在 (0, PI/2) 区间均匀选取方位角
        # MOC通常需要对称性，这里我们生成 0~90度，利用对称性覆盖 0~360
        # 权重简化处理
        for i in range(self.n_angles):
            phi = (i + 0.5) * (np.pi / 2) / self.n_angles
            self.angles.append(phi)
            self.weights.append(np.pi / 2 / self.n_angles) # 权重归一化对应 PI/2

    def _perform_ray_tracing(self):
        """核心难点：几何径迹追踪 (Ray Tracing)"""
        print("正在进行几何径迹追踪 (Ray Tracing)...")
        ray_spacing = 0.1 # 射线间距 (cm)，越小越准但越慢
        
        self.tracks = []
        
        # 针对每个角度进行追踪
        for i_ang, phi in enumerate(self.angles):
            # 方向向量
            omega_x = np.cos(phi)
            omega_y = np.sin(phi)
            
            # 为了简化，我们只处理第一象限 (0~90度) 的射线
            # 然后利用几何对称性，认为中子在四个方向是对称流动的
            # (注意：这是一个针对均匀/对称问题的简化，严谨的MOC需要处理4个象限)
            
            # 投影面积 (用于体积守恒修正)
            Ax = self.L * np.sin(phi)
            Ay = self.L * np.cos(phi)
            
            # 确定射线数量
            n_rays = int((Ax + Ay) / ray_spacing) + 1
            eff_spacing = (Ax + Ay) / n_rays # 调整后的间距以填满区域
            
            # 从左下角开始扫描
            # 坐标系旋转，我们将所有射线看作垂直于“投影面”射入
            for r in range(n_rays):
                track = Track(None, i_ang, eff_spacing)
                
                # 确定射线的起始点 (x, y)
                # 这是一个几何技巧：在旋转坐标系下均匀分布，然后逆变换回 (x,y)
                # 这里使用一种简化的“穿墙法”：
                
                # 射线的截距式： -x*sin(phi) + y*cos(phi) = dist
                dist = (r + 0.5) * eff_spacing - Ax # 偏移量
                
                # 寻找进入点 (Inlet)
                # 射线一定从 左边界(x=0) 或 下边界(y=0) 进入 (因为 phi在0-90度)
                # y = x * tan(phi) + c
                
                # ...为了代码可读性，我们采用更直观的“步进法”...
                # 我们不从外部射入，而是遍历网格边界作为起点。
                pass 

        # === 重新实现的简单版 Ray Tracing (步进法) ===
        # 上面的通用法太复杂，我们用针对笛卡尔网格的特定算法
        self.tracks = []
        for i_ang, phi in enumerate(self.angles):
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)
            tan_phi = np.tan(phi)
            
            # 射线间距在 x 和 y 轴上的投影
            delta_x = ray_spacing / sin_phi
            delta_y = ray_spacing / cos_phi
            
            # 1. 从左边界 (x=0) 出发的射线
            y_starts = np.arange(delta_y/2, self.L, delta_y)
            for y0 in y_starts:
                self._trace_single_ray(0, y0, cos_phi, sin_phi, i_ang, ray_spacing)
                
            # 2. 从下边界 (y=0) 出发的射线
            x_starts = np.arange(delta_x/2, self.L, delta_x)
            for x0 in x_starts:
                self._trace_single_ray(x0, 0, cos_phi, sin_phi, i_ang, ray_spacing)
                
        print(f"追踪完成，共生成 {len(self.tracks)} 条特征线。")

    def _trace_single_ray(self, x0, y0, dx_dir, dy_dir, angle_idx, spacing):
        """追踪单条射线，记录它切过了哪些网格"""
        track = Track((x0, y0), angle_idx, spacing)
        
        curr_x, curr_y = x0, y0
        
        # 防止死循环的安全计数
        step = 0
        max_step = self.N * 3
        
        while 0 <= curr_x < self.L and 0 <= curr_y < self.L and step < max_step:
            # 当前所在的网格索引
            ix = int(curr_x // self.dx)
            iy = int(curr_y // self.dx)
            
            # 边界修正 (防止浮点误差导致索引越界)
            if ix == self.N: ix -= 1
            if iy == self.N: iy -= 1
            
            # 计算这一步要走的距离
            # 距离右边界的距离 / x方向分量
            dist_x_wall = ((ix + 1) * self.dx - curr_x) / dx_dir
            # 距离上边界的距离 / y方向分量
            dist_y_wall = ((iy + 1) * self.dx - curr_y) / dy_dir
            
            # 谁近走谁
            segment_len = 0.0
            if dist_x_wall < dist_y_wall:
                segment_len = dist_x_wall
                # 移动坐标
                curr_x = (ix + 1) * self.dx + 1e-10 # 微小偏移防止卡在边界
                curr_y += segment_len * dy_dir
            else:
                segment_len = dist_y_wall
                curr_x += segment_len * dx_dir
                curr_y = (iy + 1) * self.dx + 1e-10
            
            track.segments.append(Segment((ix, iy), segment_len))
            step += 1
            
        self.tracks.append(track)

    def solve(self):
        """主求解器：源迭代 + 幂迭代"""
        print("开始 MOC 输运计算...")
        
        for outer in range(1000): # 外迭代 (Power Iteration)
            phi_old = self.flux.copy()
            
            # 1. 计算总源项 Q (各向同性)
            # Q = (Sigma_s * phi + 1/k * nu_Sigma_f * phi) / (4*pi)
            # 注意：MOC中习惯将 4*pi 归一化处理，这里简化处理
            source_term = (self.sigma_s * self.flux + 
                          (self.nu_sigma_f / self.k_eff) * self.flux) / (4 * np.pi)
            
            # 2. 内迭代 (Flux Sweep) - MOC 扫描
            # 初始化新的标量通量累加器
            new_flux = np.zeros_like(self.flux)
            
            # 遍历所有角度
            for i_ang in range(self.n_angles):
                # 4*pi 的权重因子 (因为我们只算了第一象限，假设4象限对称)
                # 积分 weight * flux，4个象限 x 权重
                angular_weight = self.weights[i_ang] * 4 * np.pi 
                
                # 遍历该角度下的所有射线
                # 筛选出当前角度的 track (实际优化应预先分组)
                current_tracks = [t for t in self.tracks if t.angle_idx == i_ang]
                
                for track in current_tracks:
                    # 真空边界条件：入射角通量 = 0
                    psi_in = 0.0 
                    
                    # --- 正向扫描 (Forward Sweep) ---
                    for seg in track.segments:
                        ix, iy = seg.region_idx
                        l = seg.length
                        sig_t = self.sigma_t[ix, iy]
                        q_source = source_term[ix, iy]
                        
                        # MOC 核心方程 (指数衰减)
                        # psi_out = psi_in * exp(-sig_t * l) + (Q/sig_t)*(1 - exp(-sig_t * l))
                        exp_val = np.exp(-sig_t * l)
                        psi_avg = (q_source / sig_t) + (psi_in - (q_source / sig_t)) * (1 - exp_val) / (sig_t * l)
                        psi_out = psi_in * exp_val + (q_source / sig_t) * (1 - exp_val)
                        
                        # 累加标量通量贡献
                        # contribution = psi_avg * volume_fraction
                        # 实际公式：Phi += 4*pi * weight * (Area * l / Volume) * psi_avg
                        # Area = spacing, Volume = dx*dx
                        # 简化理解：我们对每一条线段上的平均通量进行加权平均
                        
                        # 体积加权系数 = (spacing * l) / (dx * dx)
                        vol_weight = (track.spacing * l) / (self.dx * self.dx)
                        new_flux[ix, iy] += angular_weight * psi_avg * vol_weight
                        
                        # 传递通量
                        psi_in = psi_out
            
            # 更新通量
            self.flux = new_flux
            
            # 3. 更新 k_eff
            # k_new = k_old * (Total Production New / Total Production Old)
            prod_new = np.sum(self.nu_sigma_f * self.flux)
            prod_old = np.sum(self.nu_sigma_f * phi_old)
            
            k_new = self.k_eff * (prod_new / prod_old)
            k_diff = abs(k_new - self.k_eff)
            self.k_eff = k_new
            
            # 归一化
            self.flux = self.flux / np.mean(self.flux)
            
            if outer % 5 == 0:
                print(f"Iter {outer}: k_eff = {self.k_eff:.6f}, diff = {k_diff:.2e}")
            
            if k_diff < 1e-5:
                print(f"收敛! 最终 k_eff = {self.k_eff:.6f}")
                break

    def plot_flux(self):
        """画出二维通量图"""
        plt.figure(figsize=(8, 7))
        # extent设置坐标轴范围
        plt.imshow(self.flux.T, origin='lower', extent=[0, self.L, 0, self.L], cmap='jet')
        plt.colorbar(label='Normalized Scalar Flux')
        plt.title(f'2D MOC Flux Distribution\n k_eff = {self.k_eff:.5f}')
        plt.xlabel('x (cm)')
        plt.ylabel('y (cm)')
        plt.show()

# === 主程序 ===
if __name__ == "__main__":
    # 定义一个 100cm x 100cm 的反应堆
    # 网格 20x20，角度划分 8 个方位角
    solver = MOC_Solver_2D(side_length=100.0, n_mesh=40, n_angles=8)
    
    # 可以在这里修改截面做 Checkerboard 测试
    # 例如：在中心区域设置高吸收区 (控制棒)
    # mid = 20
    # solver.sigma_t[mid-5:mid+5, mid-5:mid+5] = 1.0 # 吸收截面变大
    
    solver.solve()
    solver.plot_flux()