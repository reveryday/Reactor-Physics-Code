import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================
# MOC 求解器类 (逻辑保持不变，通用引擎)
# ==========================================
class MOC_Solver_C5G7:
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
        
        print(f">> [Init] C5G7 Geometry: Pitch={pitch}cm, Radius={radius}cm")
        self.tracks, self.moc_vol_raw = self._generate_tracks()
        
        # 4. 体积校正
        self.moc_geo_area = self.moc_vol_raw * (2.0 / np.pi)
        print(f"   Geo Area: {self.volumes}")
        print(f"   MOC Area: {self.moc_geo_area}")
        
        self.boundary_psi = np.zeros((len(self.tracks), 2))

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
            
            # 正向
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
            
            # 反向
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
        print(">> [Solve] C5G7 Benchmark Start...")
        
        prod = sum(self.mats[i]['nusf']*self.flux[i]*self.volumes[i] for i in range(2))
        if prod > 0: self.flux /= prod
        
        self.k_history.append(self.k_eff)
        
        for it in range(1, 1500):
            # Source Update
            for i in range(2):
                mat = self.mats[i]
                self.source[i] = (1.0/self.k_eff)*mat['nusf']*self.flux[i] + mat['ss']*self.flux[i]
            
            # Sweep
            new_flux = self.transport_sweep()
            
            # k Update
            prod_new = sum(self.mats[i]['nusf']*new_flux[i]*self.volumes[i] for i in range(2))
            k_new = self.k_eff * prod_new
            self.k_history.append(k_new)
            
            # Norm
            if prod_new > 0:
                new_flux /= prod_new
            
            # Check
            err_k = abs(k_new - self.k_eff) / self.k_eff
            self.k_eff = k_new
            self.flux = new_flux
            
            if it % 10 == 0 or it == 1:
                ratio = self.flux[1]/self.flux[0]
                print(f"Iter {it:3d}: k={self.k_eff:.5f} | Err={err_k:.2e} | Mod/Fuel Ratio={ratio:.3f}")
            
            if err_k < 1e-5:
                print("\n>> 收敛成功 (Converged)!")
                print(f"   Calculated k_eff: {self.k_eff:.6f}")
                print(f"   Benchmark  k_inf: ~1.3339")
                print(f"   Difference: {(self.k_eff - 1.3339)*100000:.1f} pcm")
                return self.k_history, self.flux
                
        print(">> 达到最大迭代次数")
        return self.k_history, self.flux

# ==========================================
# 绘图函数 (C5G7 专用标注)
# ==========================================
def plot_results(k_history, flux_fuel, flux_mod):
    plt.style.use('bmh')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 图1: k_eff 收敛曲线
    ax1.plot(k_history, linewidth=2, color='#2E7D32', marker='.', markersize=2)
    # 画一条参考线
    ax1.axhline(y=1.3339, color='r', linestyle='--', alpha=0.5, label='Benchmark Reference (1.3339)')
    
    ax1.set_title(r'C5G7 Benchmark $k_{eff}$ Convergence', fontsize=12)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel(r'$k_{eff}$')
    ax1.legend()
    ax1.grid(True)
    
    final_k = k_history[-1]
    ax1.text(len(k_history)*0.5, min(k_history) + (max(k_history)-min(k_history))*0.4, 
             f'Final k = {final_k:.5f}\nRef = 1.3339', 
             bbox=dict(facecolor='white', alpha=0.9))

    # 图2: 区域通量
    ax2.add_patch(patches.Rectangle((-0.63, -0.63), 1.26, 1.26, 
                                    color='#B3E5FC', label='Moderator'))
    ax2.add_patch(patches.Circle((0, 0), 0.54, 
                                 color='#FFCCBC', label='Fuel (UO2)'))
    
    ax2.set_xlim(-0.7, 0.7)
    ax2.set_ylim(-0.7, 0.7)
    ax2.set_aspect('equal')
    ax2.set_title('Scalar Flux (C5G7 1-Group)', fontsize=12)
    
    ax2.text(0, 0, f"Fuel Flux\n{flux_fuel:.4f}", 
             ha='center', va='center', fontsize=10, fontweight='bold', color='#BF360C')
    ax2.text(0.48, 0.48, f"Mod Flux\n{flux_mod:.4f}", 
             ha='center', va='center', fontsize=10, fontweight='bold', color='#01579B')
    
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.show()

# ==========================================
# C5G7 Benchmark 输入参数
# 来源: OECD/NEA C5G7 Specification
# 物理数据: 7群数据经能谱加权后的单群等效值
# ==========================================

# 1. 几何尺寸 (C5G7 标准)
# 燃料棒半径比典型的 17x17 (0.41cm) 要粗
pitch = 1.26
radius = 0.54 

# 2. 材料截面 (1群等效)
# 数据解释:
# st (Total)   = Absorption + Scattering
# nusf (Prod)  = nu * Fission
# ss (Scatter) = Scattering
mat_fuel = {'st': 0.2030, 'nusf': 0.0700, 'ss': 0.1880} 
mat_mod  = {'st': 0.9000, 'nusf': 0.0000, 'ss': 0.8990} 

materials = {0: mat_fuel, 1: mat_mod}

# 3. 初始化求解器
# 增加射线数 n_rays=100 以确保几何体积计算足够精确（C5G7 燃料半径大，更需要精确）
moc = MOC_Solver_C5G7(pitch, radius, n_azimuthal=16, n_rays=100, material_data=materials)

# 4. 运行
real_k_history, real_flux = moc.solve()

# 5. 绘图
plot_results(real_k_history, real_flux[0], real_flux[1])