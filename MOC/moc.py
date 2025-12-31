import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class MOC_Solver_Fixed:
    def __init__(self, pitch, radius, n_azimuthal, n_rays, material_data):
        self.pitch = pitch  
        self.radius = radius 
        self.n_angles = n_azimuthal
        self.n_rays = n_rays
        self.mats = material_data
        
        # 1. 几何体积真值
        self.vol_fuel = np.pi * radius**2
        self.vol_mod = pitch**2 - self.vol_fuel
        self.volumes = np.array([self.vol_fuel, self.vol_mod])
        
        # 2. 物理量初始化
        self.flux = np.array([1.0, 1.0])
        self.source = np.zeros(2)
        self.k_eff = 1.0
        self.k_history = [] 
        self.tracks, self.moc_vol_raw = self._generate_tracks()
        
        # 4. 体积校正
        self.moc_geo_area = self.moc_vol_raw * (2.0 / np.pi)
        self.boundary_psi = np.zeros((len(self.tracks), 2))   # 边界角通量

    def _generate_tracks(self):
        tracks = []
        vol_accum = np.zeros(2)
        
        angles = np.linspace(0, np.pi/2, self.n_angles, endpoint=False) + (np.pi/2)/(2*self.n_angles)
        half_p = self.pitch / 2.0
        
        for angle in angles:
            sin_p = np.sin(angle)
            cos_p = np.cos(angle)
            max_proj = half_p * (np.abs(sin_p) + np.abs(cos_p))
            spacing = max_proj / (self.n_rays - 0.5)
            w_track = spacing * (np.pi / 2.0 / self.n_angles)
            
            for i in range(-self.n_rays, self.n_rays + 1):
                t = i * spacing
                if abs(t) >= max_proj: continue
                
                segs = []
                len_fuel = 0.0
                if abs(t) < self.radius:
                    len_fuel = 2.0 * np.sqrt(self.radius**2 - t**2)
                
                intersects = []
                for x in [-half_p, half_p]:
                    if abs(cos_p)>1e-9:
                        y=(x*sin_p-t)/cos_p
                        if abs(y)<=half_p+1e-6: intersects.append([x,y])
                for y in [-half_p, half_p]:
                    if abs(sin_p)>1e-9:
                        x=(t+y*cos_p)/sin_p
                        if abs(x)<=half_p+1e-6: intersects.append([x,y])
                
                unique = []
                for p in intersects:
                    if not any(np.linalg.norm(np.array(p)-np.array(u))<1e-5 for u in unique): unique.append(p)
                
                if len(unique)>=2:
                    dists = np.linalg.norm(np.array(unique)[:,None]-np.array(unique), axis=2)
                    total_len = np.max(dists)
                    len_mod = max(0.0, total_len - len_fuel)
                    
                    if len_fuel > 0:
                        segs = [(1, len_mod/2), (0, len_fuel), (1, len_mod/2)]
                    else:
                        segs = [(1, total_len)]
                    
                    tracks.append({'segs': segs, 'w': w_track})
                    for rid, length in segs:
                        vol_accum[rid] += length * w_track

        return tracks, vol_accum

    def transport_sweep(self):
        flux_accum = np.zeros(2)
        
        for t_idx, track in enumerate(self.tracks):
            w = track['w']
            
            # === 正向 ===
            psi = self.boundary_psi[t_idx, 1] 
            for rid, length in track['segs']:
                st = self.mats[rid]['st']
                q_ang = self.source[rid] / (4.0 * np.pi)
                
                if st > 1e-8:
                    exp_v = np.exp(-st * length)
                    psi_out = psi * exp_v + (q_ang/st) * (1.0 - exp_v)
                    psi_bar = (q_ang/st) - (psi_out - psi)/(st*length)
                else:
                    psi_out = psi + q_ang*length
                    psi_bar = 0.5*(psi+psi_out)
                
                flux_accum[rid] += psi_bar * length * w
                psi = psi_out
            self.boundary_psi[t_idx, 0] = psi
            
            # === 反向 ===
            psi = self.boundary_psi[t_idx, 0]
            for rid, length in reversed(track['segs']):
                st = self.mats[rid]['st']
                q_ang = self.source[rid] / (4.0 * np.pi)
                
                if st > 1e-8:
                    exp_v = np.exp(-st * length)
                    psi_out = psi * exp_v + (q_ang/st) * (1.0 - exp_v)
                    psi_bar = (q_ang/st) - (psi_out - psi)/(st*length)
                else:
                    psi_out = psi + q_ang*length
                    psi_bar = 0.5*(psi+psi_out)
                    
                flux_accum[rid] += psi_bar * length * w
                psi = psi_out
            self.boundary_psi[t_idx, 1] = psi
            
        new_flux = np.zeros(2)
        for i in range(2):
            if self.moc_vol_raw[i] > 0:
                avg_psi_in_track = flux_accum[i] / self.moc_vol_raw[i]
                new_flux[i] = 2.0 * np.pi * avg_psi_in_track
        
        return new_flux

    def solve(self):
        print(">> [Solve] 开始迭代...")
        
        prod = sum(self.mats[i]['nusf']*self.flux[i]*self.volumes[i] for i in range(2))
        if prod > 0: self.flux /= prod
        
        # 记录初始值
        self.k_history.append(self.k_eff)
        
        for it in range(1, 1500):
            # 1. 计算源
            for i in range(2):
                mat = self.mats[i]
                self.source[i] = (1.0/self.k_eff)*mat['nusf']*self.flux[i] + mat['ss']*self.flux[i]
            
            # 2. 输运
            new_flux = self.transport_sweep()
            
            # 3. 计算 k
            prod_new = sum(self.mats[i]['nusf']*new_flux[i]*self.volumes[i] for i in range(2))
            k_new = self.k_eff * prod_new
            
            # 【关键修改】 实时记录 k 值
            self.k_history.append(k_new)
            
            # 4. 归一化
            if prod_new > 0:
                new_flux /= prod_new
            
            # 5. 检查收敛
            err_k = abs(k_new - self.k_eff) / self.k_eff
            self.k_eff = k_new
            self.flux = new_flux
            
            if it % 10 == 0 or it == 1:
                ratio = self.flux[1]/self.flux[0]
                print(f"Iter {it:3d}: k={self.k_eff:.5f} | Err={err_k:.2e} | Mod/Fuel Ratio={ratio:.3f}")
            
            if err_k < 1e-5:
                print("\n>> 收敛成功!")
                print(f"   最终 k_eff: {self.k_eff:.6f}")
                # 【关键修改】 返回历史数据和最终通量
                return self.k_history, self.flux
                
        print(">> 达到最大迭代次数")
        return self.k_history, self.flux

def plot_results(k_history, flux_fuel, flux_mod):
    plt.style.use('bmh')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 图1: 真实的 k_eff 收敛曲线
    ax1.plot(k_history, linewidth=2, color='#E65100', marker='.', markersize=2)
    ax1.set_title(r'Eigenvalue ($k_{eff}$) Convergence (Actual)', fontsize=12)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel(r'$k_{eff}$')
    ax1.grid(True)
    
    # 标注最终值
    final_k = k_history[-1]
    ax1.text(len(k_history)*0.5, min(k_history) + (max(k_history)-min(k_history))*0.5, 
             f'Final k = {final_k:.6f}\nIters = {len(k_history)}', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

    # 图2: 区域通量
    ax2.add_patch(patches.Rectangle((-0.63, -0.63), 1.26, 1.26, 
                                    color='#B3E5FC', label='Moderator'))
    ax2.add_patch(patches.Circle((0, 0), 0.41, 
                                 color='#FFCCBC', label='Fuel'))
    
    ax2.set_xlim(-0.7, 0.7)
    ax2.set_ylim(-0.7, 0.7)
    ax2.set_aspect('equal')
    ax2.set_title('Scalar Flux Distribution (Calculated)', fontsize=12)
    
    # 标注真实的通量值
    ax2.text(0, 0, f"Fuel Flux\n{flux_fuel:.4f}", 
             ha='center', va='center', fontsize=10, fontweight='bold', color='#BF360C')
    ax2.text(0.45, 0.45, f"Mod Flux\n{flux_mod:.4f}", 
             ha='center', va='center', fontsize=10, fontweight='bold', color='#01579B')
    
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.show()

mat_fuel = {'st': 0.54, 'nusf': 0.22, 'ss': 0.45} 
mat_mod  = {'st': 1.40, 'nusf': 0.00, 'ss': 1.38} 
materials = {0: mat_fuel, 1: mat_mod}

pitch = 1.26
radius = 0.41

# 2. 初始化并求解
moc = MOC_Solver_Fixed(pitch, radius, 16, 60, materials)

# 3. 获取真实的计算结果
# 这里直接接收 solve() 返回的两个变量
real_k_history, real_flux = moc.solve()

# 4. 绘图 (传入真实数据)
print("正在绘制真实数据图表...")
plot_results(real_k_history, real_flux[0], real_flux[1])