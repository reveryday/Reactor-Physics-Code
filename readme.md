## 单能一维均匀平板裸堆计算---1_SN.py

- 单能：假定中子能量不会改变；
- 一维：中子通量$\phi(x)$只是$x$的函数，$y$和$z$方向无限大（一堵无限高、无限宽的墙）；
- 裸堆：无反射层；
- 目的：算出角通量分布$\psi$、通量分布$\phi$（通量分布就是把所有离散角度上的角通量加起来）、有效增值系数$k=\frac{\phi_n}{\phi_{n-1}}$；
- 边界条件：
  - 反射边界条件：$\phi(x,\mu)|_{x=0}=\phi(x,-\mu)|_{x=0}$；
  - 真空边界条件：$\phi(x,\mu)|_{x=X}=0$，没有飞进来的中子；

#### 1-理论

单能一维平板裸堆输运方程形式如下：
$$
\mu \frac{\partial\phi(x,\mu)}{\partial x}+\Sigma_t\phi(x,\mu)=(\frac{\Sigma_s(x)}{2}+\frac{\nu\Sigma_f}{2k_{eff}})\int_{-1}^1\phi(x,\mu')d\mu'
$$
其中，$\mu=\cos{\theta}\in[-1,1]$。定义标量通量$\phi(x)=\int_{-1}^1\phi(x,\mu')d\mu'$，则：
$$
\mu \frac{\partial\phi(x,\mu)}{\partial x}+\Sigma_t\phi(x,\mu)=(\frac{\Sigma_s(x)}{2}+\frac{\nu\Sigma_f}{2k_{eff}})\phi(x)
$$
$\phi(x)$的积分使用高斯求积可得N个联立的微分方程：
$$
\mu_m \frac{\partial\phi_m(x)}{\partial x}+\Sigma_t\phi_m(x)=(\frac{\Sigma_s(x)}{2}+\frac{\nu\Sigma_f}{2k_{eff}})\sum_{j=1}^Nw_j\phi_j(x)
$$
其中，$m=1,2,...,N$，$\phi_m(x)$代表角通量-在x处角度为$\mu_m$的通量；上式左边是角度为$\mu_m$坐标为x的通量变化和移出量，右边是其他角度由于散射、裂变到$\mu_m$得到的源项，散射与裂变均视为各向同性的。

（注：高斯-勒让德求积的求积节点$\mu_1,\mu_2,...\mu_N$与对应的权$w_m$可直接生成，将$\mu_m$代入$\phi(x,\mu)$即可得到$\phi_m(x)$，这里的N也就是SN方法角度离散的个数。）

再通过划分成K个网格，$\phi(x,\mu)$就是一个$K\times N$的二维矩阵，得到差分方程：
$$
\mu_m\frac{\phi_{k+1/2,m}-\phi_{k-1/2,m}}{\Delta x_k}+\Sigma_t\phi_{k,m}=S_k
$$
其中$m=1,2,...,N,\ k=1,2,...,K$。（通俗来讲就是，在某一个k网格内，$\frac{\psi_R-\phi_L}{\Delta x}$视作k处的psi随x的变化率，$S_k$是k处的源项）

该差分方程可写为：
$$
\mu\frac{\psi_R-\psi_L}{\Delta x}+\Sigma_t\phi_{center}=Q
$$
其中$\psi_{center}=\frac{1}{2}(\psi_R+\psi_L)$，从右向左扫描时可得：
$$
\psi_L(\frac{|\mu|}{\Delta x}+\frac{\Sigma_t}{2})=\psi_R(\frac{|\mu|}{\Delta x}-\frac{\Sigma_t}{2})+\frac{Q}{2}
$$

其中：
$$
c = \frac{\Sigma_s + \nu\Sigma_f}{\Sigma_t} = \frac{0.030 + 0.0225}{0.050} = \frac{0.0525}{0.050} = 1.05
$$

#### 2-内迭代与外迭代

- 1_SN.py代码中的phi代表：标量通量$\phi(x)$，单位体积内，所有中子运动轨迹的总长度；
- 1_SN.py代码中的psi代表-角通量$\phi(x,\mu)$，单位体积内，能量为 $E$，且沿 $\mu$ 方向运动的中子的总轨迹长度；

流程：先猜一个phi，得到裂变源项source_fission，【计算散射源，基于总源项计算扫描得到psi，再把psi对角度求积得到phi_new，与phi比较直到其收敛，于是得到下一代phi】（内迭代），再将这个phi代入得到**新一代**的裂变项再内迭代，直到diff_k < tolerance_k即收敛则停止，得到最终的psi、phi；

- 外迭代：假定k和一个裂变源；

- 内迭代：已知phi，固定裂变源，只处理散射，更新得到当代稳定后的phi；

## 单能一维MC---1_mc.py

#### 1-理论

中子在介质中飞行而不发生碰撞的概率，服从指数衰减律，中子恰好飞行距离 $x$ 后发生碰撞的概率是：
$$
p(x) = \Sigma_t e^{-\Sigma_t x}
$$
