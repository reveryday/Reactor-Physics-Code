import numpy as np
import matplotlib.pyplot as plt

class MOC_2D_PinCell:
    def __init__(self):
        # ==============================
        # 1. 几何与物理参数
        # ==============================
        self.pitch = 1.26         # 栅距 (cm)
        self.radius = 0.4096      # 燃料棒半径 (cm) (典型压水堆尺寸)
        
        # 区域定义: 0=Fuel, 1=Moderator
        self.n_regions = 2
        
        # 真实几何体积 (cm^2)
        self.vol_geo = np.zeros(self.n_regions)
        self.vol_geo[0] = np.pi * self.radius**2
        self.vol_geo[1] = self.pitch**2 - self.vol_geo[0]
        
        # 物理截面 (单群)
        # Fuel: 高吸收, 高裂变 (nu*Sf > Sa 才有 k>1)
        # k_inf_fuel = 0.7 / 0.4 = 1.75
        self.mat_fuel = {'St': 1.0, 'nSf': 0.7, 'Sa': 0.4} 
        self.mat_fuel['Ss'] = self.mat_fuel['St'] - self.mat_fuel['Sa']
        
        # Mod: 低吸收, 纯散射
        self.mat_mod = {'St': 1.0, 'nSf': 0.0, 'Sa': 0.02}
        self.mat_mod['Ss'] = self.mat_mod['St'] - self.mat_mod['Sa']
        
        self.materials = [self.mat_fuel, self.mat_mod]
        
        # MOC 参数
        self.n_azimuthal = 16   # 方位角数量 (0~PI)
        self.ray_spacing = 0.03 # 射线间距 (cm)
        
        # 求解变量
        self.flux = np.ones(self.n_regions)
        self.source = np.zeros(self.n_regions)
        self.keff = 1.0
        
        # 边界通量 (入射角通量)
        self.boundary_psi = 1.0

    def generate_tracks(self):
        """
        生成特征线轨迹 (Ray Tracing)
        """
        print(">>> 生成特征线轨迹...")
        self.tracks = []
        
        # 在 [0, pi] 上均匀取角
        angles = np.linspace(0, np.pi, self.n_azimuthal, endpoint=False) + (np.pi/self.n_azimuthal)/2
        
        for angle in angles:
            sin_a = np.sin(angle)
            cos_a = np.cos(angle)
            
            # 计算投影宽度 (旋转后的矩形宽度)
            # 投影宽度 = |w*cos| + |h*sin|
            proj_width = abs(self.pitch * cos_a) + abs(self.pitch * sin_a)
            
            # 生成射线偏移量
            n_rays = int(proj_width / self.ray_spacing) + 1
            offsets = np.linspace(-proj_width/2, proj_width/2, n_rays)
            
            # 实际的射线间距 (微调以填满宽度)
            real_spacing = proj_width / n_rays if n_rays > 0 else self.ray_spacing
            
            for b in offsets:
                # 追踪单条射线
                segments = self._trace_single_ray(angle, b)
                
                if len(segments) > 0:
                    # 权重 = 射线间距 * 角度权重
                    # 角度权重 = pi / N_angles (均匀分布)
                    weight = real_spacing * (np.pi / self.n_azimuthal)
                    
                    self.tracks.append({
                        'segs': segments,
                        'w': weight
                    })
        
        print(f"    共生成 {len(self.tracks)} 条轨迹")
        self._correct_volumes()

    def _trace_single_ray(self, angle, b):
        """计算直线与 Fuel/Mod 的交点"""
        P2 = self.pitch / 2.0
        sin_a, cos_a = np.sin(angle), np.cos(angle)
        
        # 1. 计算与矩形边界的交点
        intersects = []
        
        # x = +/- P2
        if abs(cos_a) > 1e-9:
            for x in [-P2, P2]:
                y = (b + x * sin_a) / cos_a
                if -P2-1e-5 <= y <= P2+1e-5:
                    intersects.append((x, y))
        # y = +/- P2
        if abs(sin_a) > 1e-9:
            for y in [-P2, P2]:
                x = (y * cos_a - b) / sin_a
                if -P2-1e-5 <= x <= P2+1e-5:
                    intersects.append((x, y))
        
        # 去重并排序
        valid_pts = sorted(list(set([(round(p[0],6), round(p[1],6)) for p in intersects])),
                           key=lambda p: p[0]*cos_a - p[1]*sin_a) # 沿垂直于射线的方向排序有问题，应沿射线方向
        # 修正排序：沿射线方向 t = x*sin - y*cos (注意坐标系定义)
        # 直线方程: x*sin - y*cos = -b. 参数方程 x=x0 - t*cos, y=y0 - t*sin ??
        # 简单点：按 x 或 y 排序即可，只要单调
        if abs(cos_a) > abs(sin_a):
            valid_pts.sort(key=lambda p: p[0] if cos_a>0 else -p[0])
        else:
            valid_pts.sort(key=lambda p: p[1] if sin_a>0 else -p[1])

        if len(valid_pts) < 2: return []
        
        p_start, p_end = valid_pts[0], valid_pts[-1]
        total_len = np.hypot(p_end[0]-p_start[0], p_end[1]-p_start[1])
        if total_len < 1e-6: return []

        # 2. 计算与圆的交点
        # 直线到圆心(0,0)距离 = |b| (因为我们构建方程时 b就是截距)
        # 注意：这里的几何构建稍微有点tricky，上面的投影法构建中，b就是距离
        # 验证：直线方程 -x*sin + y*cos = b (或者类似的旋转)。
        # 在 generate_tracks 里，我们实际上是把坐标系旋转了 angle。
        # 在旋转坐标系下，y' = b 是直线。
        # 圆心在 (0,0)。距离确实是 |b|。
        
        dist = abs(b)
        segs = []
        
        if dist < self.radius:
            half_chord = np.sqrt(self.radius**2 - dist**2)
            l_fuel = 2 * half_chord
            l_mod = max(0, total_len - l_fuel)
            
            # 简化模型：认为燃料总在中间
            # Mod -> Fuel -> Mod
            segs.append({'id': 1, 'l': l_mod/2})
            segs.append({'id': 0, 'l': l_fuel})
            segs.append({'id': 1, 'l': l_mod/2})
        else:
            # 只有慢化剂
            segs.append({'id': 1, 'l': total_len})
            
        return segs

    def _correct_volumes(self):
        """
        关键步骤：体积修正
        MOC积分出的体积 sum(w*l) 必须等于真实几何体积，否则通量归一化会出错
        """
        vol_moc = np.zeros(self.n_regions)
        for t in self.tracks:
            for s in t['segs']:
                vol_moc[s['id']] += s['l'] * t['w']
        
        print(f"    几何体积: {self.vol_geo}")
        print(f"    MOC 体积: {vol_moc}")
        
        # 计算修正因子
        self.vol_corr = np.divide(self.vol_geo, vol_moc, out=np.ones_like(vol_moc), where=vol_moc!=0)
        print(f"    体积修正因子: {self.vol_corr}")
        
        # 将修正因子乘到每条轨迹的长度或权重上，这里乘到权重上方便后续计算
        for t in self.tracks:
            # 注意：一条轨迹可能穿过多个区域，权重是共享的。
            # 所以不能简单修改 t['w']。
            # 我们在积分通量时，动态应用修正因子。
            pass 

    def solve(self, max_iter=200, tol=1e-5):
        print("\n>>> 开始源迭代...")
        
        for it in range(max_iter):
            # 1. 计算源项 Q (各向同性)
            # Source = (1/k)*nuSf*Phi + Ss*Phi
            # MOC 扫描使用的是 q = Q / (4*pi)
            for r in range(self.n_regions):
                mat = self.materials[r]
                self.source[r] = (1.0/self.keff)*mat['nSf']*self.flux[r] + mat['Ss']*self.flux[r]
            
            # 2. 输运扫描 (MOC Sweep)
            # 累加量
            new_flux_num = np.zeros(self.n_regions) # 分子: sum(w * l * psi_avg)
            
            total_leakage = 0.0 # 飞出去的总权重
            total_weight = 0.0  # 总权重
            
            for t in self.tracks:
                # 入射通量 (白边界：各向同性反射)
                psi = self.boundary_psi
                
                for s in t['segs']:
                    rid = s['id']
                    length = s['l']
                    mat = self.materials[rid]
                    st = mat['St']
                    vol_factor = self.vol_corr[rid] # 关键：应用体积修正
                    
                    # 标量源 -> 角源
                    q_ang = self.source[rid] / (4*np.pi)
                    
                    # 解析解
                    # psi_out = psi_in * exp(-St*L) + (q/St)*(1 - exp(-St*L))
                    if st > 1e-6:
                        exp_f = np.exp(-st * length)
                        psi_out = psi * exp_f + (q_ang/st) * (1.0 - exp_f)
                        psi_avg = (q_ang/st) - (psi_out - psi) / (st * length)
                    else:
                        psi_out = psi + q_ang * length
                        psi_avg = psi + 0.5 * q_ang * length
                        
                    # 积分通量：Phi * V = sum( 4pi * w * l * psi_avg )
                    # 我们这里累加的是 Phi * V 的一部分
                    # 引入 vol_factor 确保 volume consistency
                    
                    # 权重 w 已经包含了 delta_r 和 delta_angle
                    # 在 2D 中，通常对 2pi 积分。如果是 4pi，系数相应变化。
                    # 这里简化处理：我们只关心相对比例，最后会归一化。
                    
                    term = t['w'] * length * psi_avg * vol_factor
                    new_flux_num[rid] += term
                    
                    # 传递到下一段
                    psi = psi_out
                
                # 收集出射通量用于边界更新
                total_leakage += psi * t['w']
                total_weight += t['w']
                
            # 3. 更新边界条件 (White BC)
            # 下一代的入射 = 这一代的出射平均值
            if total_weight > 0:
                self.boundary_psi = total_leakage / total_weight
                
            # 4. 更新标量通量
            # Phi = (Sum term) / Vol_geo * 4pi
            for r in range(self.n_regions):
                # 4pi 因子来自于将角通量积分为标量通量
                # 由于我们在 MOC 公式里除以了 4pi 算 q_ang，这里要乘回来
                self.flux[r] = (new_flux_num[r] * 4 * np.pi) / self.vol_geo[r]
            
            # 5. !!! 通量重归一化 (Flux Renormalization) !!!
            # 防止通量因 k!=1 而归零或爆炸
            # 强行将平均通量设为 1.0
            avg_flux = np.sum(self.flux * self.vol_geo) / np.sum(self.vol_geo)
            self.flux /= avg_flux
            
            # 6. 计算 k_eff
            # k_new = k_old * (Production_New / Production_Old)
            # Production = sum(nuSf * Phi * V)
            # 由于我们对 Phi 做了归一化，我们不能直接比较 Source 变化。
            # 标准做法：k = Total_Production / Total_Loss
            # Total Loss = Absorption + Leakage (Boundary is reflective, so Leakage=0)
            # Total Loss = sum(Sa * Phi * V)
            
            prod_rate = sum([m['nSf'] * self.flux[i] * self.vol_geo[i] for i, m in enumerate(self.materials)])
            abs_rate  = sum([m['Sa']  * self.flux[i] * self.vol_geo[i] for i, m in enumerate(self.materials)])
            
            # k = Production / Absorption (在无泄露情况下)
            k_new = prod_rate / abs_rate
            
            # 检查收敛
            err = abs(k_new - self.keff)
            self.keff = k_new
            
            if (it+1) % 5 == 0:
                print(f"Iter {it+1:3d}: k_eff = {self.keff:.5f}, Fuel Flux = {self.flux[0]:.3f}, Mod Flux = {self.flux[1]:.3f}")
            
            if err < tol:
                print(f"\n>>> 收敛于第 {it+1} 步")
                break
                
        return self.keff, self.flux

def plot_moc_3d_distribution(solver):
    """
    绘制 MOC 计算结果的 3D 通量分布图
    """
    print("正在生成 3D 可视化...")
    
    # 1. 创建高分辨率网格 (为了画出圆形的轮廓)
    resolution = 200
    p_half = solver.pitch / 2.0
    x = np.linspace(-p_half, p_half, resolution)
    y = np.linspace(-p_half, p_half, resolution)
    X, Y = np.meshgrid(x, y)
    
    # 2. 重建通量场
    # MOC 的结果是区域平均值，所以我们需要把值"填"回几何网格里
    Z = np.zeros_like(X)
    
    # 计算每个点到中心的距离
    R_dist = np.sqrt(X**2 + Y**2)
    
    # 获取计算出的通量值
    val_fuel = solver.flux[0]
    val_mod  = solver.flux[1]
    
    # 赋值：圆内赋 Fuel 值，圆外赋 Mod 值
    # 使用 NumPy 的掩码操作
    mask_fuel = R_dist <= solver.radius
    Z[mask_fuel] = val_fuel
    Z[~mask_fuel] = val_mod  # 取反，即圆外
    
    # 3. 绘图
    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    # 绘制曲面
    # cmap='viridis' 颜色对比度好
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', 
                           linewidth=0, antialiased=False, alpha=0.9)
    
    # 4. 绘制底部的几何投影 (辅助线)
    # 画一个红色的圆圈表示燃料棒边界
    theta = np.linspace(0, 2*np.pi, 100)
    xc = solver.radius * np.cos(theta)
    yc = solver.radius * np.sin(theta)
    # zdir='z', offset=... 把线画在底部
    z_floor = np.min(Z) * 0.95
    ax.plot(xc, yc, zs=z_floor, zdir='z', color='red', linewidth=2, linestyle='--', label='Fuel Pin Boundary')
    
    # 5. 设置坐标轴和标题
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Neutron Flux (Arbitrary Units)')
    ax.set_zlim(z_floor, np.max(Z)*1.05)
    
    title_text = f"MOC Flux Distribution (Flat Source Approx.)\n$k_{{eff}}={solver.keff:.5f}$"
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    
    # 添加颜色条
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Flux Magnitude')
    
    # 调整视角 (俯视角度更容易看清凹陷)
    ax.view_init(elev=45, azim=-45)
    
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. 初始化并计算
    solver = MOC_2D_PinCell()
    solver.generate_tracks()
    k, phi = solver.solve()
    
    # 2. 打印数值结果
    print("="*40)
    print(f"Final Result: k_eff = {k:.5f}")
    print(f"Fuel Flux      = {phi[0]:.4f}")
    print(f"Moderator Flux = {phi[1]:.4f}")
    print("="*40)
    
    # 3. 画图 (直接调用上面的函数)
    plot_moc_3d_distribution(solver)