import numpy as np
import matplotlib.pyplot as plt
import time

""" 一维平板裸堆 MOC 求解（可跟一维蒙卡程序互相验证） """

class SlabMOC:
    def __init__(self, length, n_cells, n_angles=8):
        self.length = length  # 平板的长度
        self.n_cells = n_cells # 网格数
        self.dx = length / n_cells # 网格尺寸
        self.n_angles = n_angles # 离散角数
        self.x_centers = np.linspace(self.dx/2, self.length - self.dx/2, self.n_cells) # 网格中心坐标
        mus, weights = np.polynomial.legendre.leggauss(n_angles)  # 16维的勒让德求积点和权重
        self.mus = mus
        self.weights = weights
        
        # 初始化
        self.sigma_t = np.zeros(self.n_cells) # 总截面
        self.sigma_s = np.zeros(self.n_cells) # 散射截面
        self.nusigma_f = np.zeros(self.n_cells) # 产额截面
        self.phi = np.ones(self.n_cells) # 初始phi
        self.source = np.zeros(self.n_cells) # 初始源项
        self.k_eff = 1.0

    # 通过x坐标获取截面
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
                
            self.sigma_t[i] = sa + ss
            self.sigma_s[i] = ss
            self.nusigma_f[i] = nusf

    def transport_sweep(self):
        new_phi = np.zeros(self.n_cells) # 初始化新的phi
        
        # 遍历所有角度
        for n in range(self.n_angles):
            mu = self.mus[n]  # 角度取高斯-勒让德求积点
            w = self.weights[n] # 对应的权重
            
            q_angular = self.source / 2.0  # 角度分量
            psi_in = 0.0  # 真空边界条件
            
            if mu > 0:
                # 向右扫描-遍历每一个网格
                for i in range(self.n_cells):
                    sigma_t = self.sigma_t[i]  # 总截面
                    tau = sigma_t * self.dx / abs(mu)  # 光学厚度
                    
                    # 特征线解析解
                    if sigma_t > 1e-8:
                        exp_val = np.exp(-tau)
                        psi_out = psi_in * exp_val + (q_angular[i]/sigma_t) * (1.0 - exp_val)
                        psi_avg = (q_angular[i] - (mu/self.dx)*(psi_out - psi_in)) / sigma_t  # 当前网格内部的平均通量
                    else:
                        psi_out = psi_in + q_angular[i] * (self.dx/abs(mu))
                        psi_avg = 0.5 * (psi_in + psi_out)
                    
                    # 累加通量: Phi = Sum(w * Psi_avg)
                    new_phi[i] += psi_avg * w
                    
                    # 更新下一格的入射
                    psi_in = psi_out
                    
            else:
                # mu<0 --> 向左扫描
                for i in range(self.n_cells - 1, -1, -1):
                    st = self.sigma_t[i]
                    tau = st * self.dx / abs(mu)
                    
                    if st > 1e-8:
                        exp_val = np.exp(-tau)
                        psi_out = psi_in * exp_val + (q_angular[i]/st) * (1.0 - exp_val)
                        psi_avg = (q_angular[i]/st) - (psi_out - psi_in) * (abs(mu)/(st*self.dx)) 
                    else:
                        psi_out = psi_in + q_angular[i] * (self.dx/abs(mu))
                        psi_avg = 0.5 * (psi_in + psi_out)
                        
                    new_phi[i] += psi_avg * w
                    psi_in = psi_out
                    
        return new_phi

    def solve(self, max_iter=1000, tol=1e-6):
        start_time = time.time()    
        self.get_materials()
        
        # 归一化初始通量
        norm = np.sum(self.phi)
        self.phi /= norm
        
        # 源迭代
        for it in range(1, max_iter+1):
            self.source = (1.0/self.k_eff) * self.nusigma_f * self.phi + \
                          self.sigma_s * self.phi
            
            new_phi = self.transport_sweep() # 通过输运扫描更新通量

            # fiss_old = np.sum(self.nusigma_f * self.phi) # 裂变率加权平均
            # fiss_new = np.sum(self.nusigma_f * new_phi)
            # k_new = self.k_eff * (fiss_new / fiss_old)

            total_new = np.sum(new_phi) # 直接求总通量
            total_old = np.sum(self.phi)
            k_new = self.k_eff * (total_new / total_old)

            err_k = abs(k_new - self.k_eff) / self.k_eff
            err_phi = np.max(np.abs(new_phi - self.phi) / (self.phi + 1e-12))
            
            self.k_eff = k_new
            self.phi = new_phi
            self.phi /= np.sum(self.phi)
            
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
        plt.plot(self.x_centers, self.phi, 'b-', linewidth=2)
        
        # 画出区域分界线
        plt.axvline(x=50, color='r', linestyle='--', label='Interface')
        plt.axvspan(0, 50, color='red', alpha=0.1, label='Region 1')
        plt.axvspan(50, 100, color='green', alpha=0.1, label='Region 2')
        
        plt.xlabel('x (cm)')
        plt.ylabel('phi')
        plt.title(f'phi Distribution (1D Slab MOC)\n$k_{{eff}} = {self.k_eff:.5f}$')
        plt.legend()
        plt.grid(True)
        plt.show()

moc = SlabMOC(length=100.0, n_cells=200, n_angles=16)
moc.solve()
moc.plot_results()