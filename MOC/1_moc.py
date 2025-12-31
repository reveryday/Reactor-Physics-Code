import numpy as np
import matplotlib.pyplot as plt
import time

class SlabMOC:
    def __init__(self, length, n_cells, n_angles=8):
        self.L = length
        self.N = n_cells
        self.dx = length / n_cells
        self.n_angles = n_angles
        self.x_centers = np.linspace(self.dx/2, self.L - self.dx/2, self.N)
        mus, weights = np.polynomial.legendre.leggauss(n_angles)
        self.mus = mus
        self.weights = weights
        
        # 初始化
        self.sig_t = np.zeros(self.N)
        self.sig_s = np.zeros(self.N)
        self.nusig_f = np.zeros(self.N)
        self.flux = np.ones(self.N) # 初始猜测标量通量
        self.source = np.zeros(self.N)
        self.k_eff = 1.0

    def get_materials(self):
        for i, x in enumerate(self.x_centers):
            
            if 0 <= x < 50:
                # Region 1
                sa = 0.12
                ss = 0.05
                nusf = 0.15
            else:
                # Region 2 (50 < x < 100)
                sa = 0.10
                ss = 0.05
                nusf = 0.12
                
            self.sig_t[i] = sa + ss
            self.sig_s[i] = ss
            self.nusig_f[i] = nusf

    def transport_sweep(self):
        new_flux = np.zeros(self.N)
        
        # 遍历所有角度
        for n in range(self.n_angles):
            mu = self.mus[n]
            w = self.weights[n]
            
            q_angular = self.source / 2.0  
            psi_in = 0.0  # 真空边界条件
            
            if mu > 0:
                # ---> 向右扫描
                for i in range(self.N):
                    st = self.sig_t[i]
                    tau = st * self.dx / abs(mu) # 光学厚度
                    
                    # 特征线解析解
                    if st > 1e-8:
                        exp_val = np.exp(-tau)
                        psi_out = psi_in * exp_val + (q_angular[i]/st) * (1.0 - exp_val)
                        psi_avg = (q_angular[i] - (mu/self.dx)*(psi_out - psi_in)) / st
                    else:
                        psi_out = psi_in + q_angular[i] * (self.dx/abs(mu))
                        psi_avg = 0.5 * (psi_in + psi_out)
                    
                    # 累加标量通量: Phi = Sum(w * Psi_avg)
                    new_flux[i] += psi_avg * w
                    
                    # 更新下一格的入射
                    psi_in = psi_out
                    
            else:
                # <--- 向左扫描
                for i in range(self.N - 1, -1, -1):
                    st = self.sig_t[i]
                    tau = st * self.dx / abs(mu)
                    
                    if st > 1e-8:
                        exp_val = np.exp(-tau)
                        psi_out = psi_in * exp_val + (q_angular[i]/st) * (1.0 - exp_val)
                        psi_avg = (q_angular[i]/st) - (psi_out - psi_in) * (abs(mu)/(st*self.dx)) 
                    else:
                        psi_out = psi_in + q_angular[i] * (self.dx/abs(mu))
                        psi_avg = 0.5 * (psi_in + psi_out)
                        
                    new_flux[i] += psi_avg * w
                    psi_in = psi_out
                    
        return new_flux

    def solve(self, max_iter=1000, tol=1e-6):
        start_time = time.time()    
        self.get_materials()
        
        # 归一化初始通量
        norm = np.sum(self.flux)
        self.flux /= norm
        
        for it in range(1, max_iter+1):
            # 1. 构建源项 (Source Iteration)
            # Q = (1/k)*Fission + Scattering
            self.source = (1.0/self.k_eff) * self.nusig_f * self.flux + \
                          self.sig_s * self.flux
            
            # 输运扫描-更新k
            new_flux = self.transport_sweep()
            fiss_old = np.sum(self.nusig_f * self.flux)
            fiss_new = np.sum(self.nusig_f * new_flux)
            
            k_new = self.k_eff * (fiss_new / fiss_old)
            
            # 4. 检查收敛
            err_k = abs(k_new - self.k_eff) / self.k_eff
            err_phi = np.max(np.abs(new_flux - self.flux) / (self.flux + 1e-12))
            
            self.k_eff = k_new
            self.flux = new_flux
            
            # 每一代归一化通量防止数值溢出
            self.flux /= np.sum(self.flux)
            
            if it % 10 == 0:
                print(f"Iter {it:3d}: k_eff = {self.k_eff:.6f} | Err = {err_k:.2e}")
                
            if err_k < tol and err_phi < tol:
                end_time = time.time()
                print(f"最终 k_eff: {self.k_eff:.6f}")
                print(f"计算耗时: {end_time - start_time:.2f} 秒")
                return

        print(">> 达到最大迭代次数，未收敛。")

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.x_centers, self.flux, 'b-', linewidth=2)
        
        # 画出区域分界线
        plt.axvline(x=50, color='r', linestyle='--', label='Interface')
        plt.axvspan(0, 50, color='red', alpha=0.1, label='Region 1')
        plt.axvspan(50, 100, color='green', alpha=0.1, label='Region 2')
        
        plt.xlabel('x (cm)')
        plt.ylabel('Flux')
        plt.title(f'Flux Distribution (1D Slab MOC)\n$k_{{eff}} = {self.k_eff:.5f}$')
        plt.legend()
        plt.grid(True)
        plt.show()

moc = SlabMOC(length=100.0, n_cells=200, n_angles=16)
moc.solve()
moc.plot_results()