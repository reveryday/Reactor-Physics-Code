import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_reactor_geometry():
    # 创建画布
    fig, ax = plt.subplots(figsize=(8, 8))

    # ==============================
    # 1. 绘制整个反应堆区域 (Uniform Medium)
    # ==============================
    # 矩形: 左下角(0,0), 宽100, 高100
    reactor_rect = patches.Rectangle(
        (0, 0), 100, 100, 
        linewidth=2, 
        edgecolor='black', 
        facecolor='#e0e0e0', # 浅灰色代表均匀介质
        label='Uniform Medium'
    )
    ax.add_patch(reactor_rect)

    # ==============================
    # 2. 绘制源区域 (Source Region)
    # ==============================
    # 题目条件: 0 < x < 25, 25 < y < 50
    # 左下角坐标 (0, 25), 宽 25, 高 25 (50-25)
    source_rect = patches.Rectangle(
        (0, 25), 25, 25, 
        linewidth=2, 
        edgecolor='#d62728', # 深红色边框
        facecolor='#ff7f0e', # 橙色填充
        alpha=0.7,           # 半透明
        label='Source Region'
    )
    ax.add_patch(source_rect)

    # ==============================
    # 3. 添加文字标注
    # ==============================
    # 标注源
    ax.text(12.5, 37.5, 'Source\n$S_0$', 
            horizontalalignment='center', verticalalignment='center', 
            fontsize=12, fontweight='bold', color='white')

    # 标注介质区域
    ax.text(60, 50, '$\Sigma_t=0.25, \Sigma_{s0}=0.15$', 
            horizontalalignment='center', verticalalignment='center', 
            fontsize=12, color='black')
            
    # 标注真空边界
    ax.text(50, -5, 'Vacuum Boundary', ha='center', fontsize=10, fontstyle='italic')
    ax.text(-5, 50, 'Vacuum\nBoundary', va='center', ha='right', fontsize=10, fontstyle='italic', rotation=90)
    ax.text(50, 105, 'Vacuum Boundary', ha='center', fontsize=10, fontstyle='italic')
    ax.text(105, 50, 'Vacuum\nBoundary', va='center', ha='left', fontsize=10, fontstyle='italic', rotation=270)

    # ==============================
    # 4. 图表设置
    # ==============================
    # 设置坐标轴范围 (留出一点边距)
    ax.set_xlim(-15, 115)
    ax.set_ylim(-15, 115)
    
    # 设置坐标轴标签
    ax.set_xlabel('x (cm)', fontsize=12)
    ax.set_ylabel('y (cm)', fontsize=12)
    ax.set_title('Problem 9.12 Reactor Geometry', fontsize=14, pad=20)
    
    # 开启网格，方便看坐标
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 保持 x 和 y 轴比例一致 (正方形看起来要是正方形)
    ax.set_aspect('equal')
    
    # 显示图例
    ax.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_reactor_geometry()