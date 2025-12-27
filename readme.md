## 单能一维均匀平板裸堆计算---1_SN.py

- 一维：中子通量$\phi(x)$只是$x$的函数，$y$和$z$方向无限大（想象一堵无限高、无限宽的墙）；
- 裸堆：无反射层；
- 目的：算出角通量分布$\psi$、通量分布$\phi$（通量分布就是把所有离散角度上的角通量加起来）、有效增值系数$k=\frac{\phi_n}{\phi_{n-1}}$；
- 边界条件：
  - 反射边界条件：$\phi(x,\mu)|_{x=0}=\phi(x,-\mu)|_{x=0}$；
  - 真空边界条件：$\phi(x,\mu)|_{x=X}=0$；

单能一维平板裸堆输运方程形式如下：
$$
\mu \frac{\partial\phi(x,\mu)}{\partial x}+\Sigma_t\phi(x,\mu)=(\frac{\Sigma_s(x)}{2}+\frac{\nu\Sigma_f}{2k_{eff}})\int_{-1}^1\phi(x,\mu')d\mu'
$$
其中，$\mu=\cos{\theta}\in[-1,1]$。定义标量通量$\phi(x)=\int_{-1}^1\phi(x,\mu')d\mu'$，则：
$$
\mu \frac{\partial\phi(x,\mu)}{\partial x}+\Sigma_t\phi(x,\mu)=(\frac{\Sigma_s(x)}{2}+\frac{\nu\Sigma_f}{2k_{eff}})\phi(x)
$$
$\phi(x)$的积分使用高斯求积可得：
$$
\mu \frac{\partial\phi(x,\mu)}{\partial x}+\Sigma_t\phi(x,\mu)=(\frac{\Sigma_s(x)}{2}+\frac{\nu\Sigma_f}{2k_{eff}})\sum_{m=1}^Nw_m\phi_m(x)
$$
（注：高斯-勒让德求积的求积节点$\mu_1,\mu_2,...\mu_N$与对应的权$w_m$可直接生成，将$\mu_m$代入$\phi(x,\mu)$即可得到$\phi_m(x)$，这里的N也就是SN方法角度离散的个数。）

#### 1-内迭代与外迭代

- 1_SN.py代码中的phi代表：标量通量$\phi(x)$；
- 1_SN.py代码中的psi代表-角通量$\phi(x,\mu)$；

外迭代：

内迭代：已知上一代通量phi，固定裂变源，更新标量通量$\phi(x)$的过程，本质上就是把所有方向上的角通量$\phi(x.\mu)$加起来。

1. 先计算得到一个初始源项；
2. 负方向+正方向扫描；

#### 2-菱形差分公式

建立立网格中心通量与网格边缘通量之间的数学关系，从而使方程组闭合可解。菱形差分假设：网格中心的通量，是左边界和右边界通量的算术平均值：
$$
\psi_{center} = \frac{1}{2} (\psi_{L} + \psi_{R})
$$









