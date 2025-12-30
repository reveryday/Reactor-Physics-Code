import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_mc_slab_geometry():
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    
    # ==============================
    # 1. 绘制平板区域 (Slab Regions)
    # ==============================
    # 虽然是1D，但在图中画成有一定高度的矩形条更直观
    # 高度设为 1.0 (任意单位)
    
    # --- Region 1 (0 - 50 cm) ---
    rect_r1 = patches.Rectangle((0, 0), 50, 1, 
                                linewidth=2, edgecolor='black', facecolor='#A0C4FF', 
                                label='Region 1\n(Material 1)')
    ax.add_patch(rect_r1)
    
    # --- Region 2 (50 - 100 cm) ---
    rect_r2 = patches.Rectangle((50, 0), 50, 1, 
                                linewidth=2, edgecolor='black', facecolor='#FFADAD', 
                                label='Region 2\n(Material 2)')
    ax.add_patch(rect_r2)

    # ==============================
    # 2. 标注尺寸 (Dimensions)
    # ==============================
    
    # 辅助函数：画双箭头标注 (大字体版)
    def add_dim_arrow(x1, y, x2, text, text_offset=-0.15):
        ax.annotate('', xy=(x1, y), xytext=(x2, y),
                    arrowprops=dict(arrowstyle='<->', lw=2))
        mid_x = (x1 + x2) / 2
        ax.text(mid_x, y + text_offset, text, va='top', ha='center', fontsize=14)

    # 标注 Region 1 宽度
    add_dim_arrow(0, -0.1, 50, '50 cm')
    # 标注 Region 2 宽度
    add_dim_arrow(50, -0.1, 100, '50 cm')
    # 标注 总宽度
    add_dim_arrow(0, -0.4, 100, 'Total Thickness: 100 cm', text_offset=-0.15)

    # ==============================
    # 3. 标注边界条件与界面
    # ==============================
    
    # --- 真空边界 (Vacuum Boundary) ---
    # 左边界
    ax.annotate('Vacuum\nBoundary', xy=(0, 0.5), xytext=(-15, 0.5),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=14, ha='center', va='center')
    # 右边界
    ax.annotate('Vacuum\nBoundary', xy=(100, 0.5), xytext=(115, 0.5),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=14, ha='center', va='center')
    
    # --- 界面 (Interface) ---
    ax.axvline(x=50, color='black', linestyle='--', linewidth=1.5, ymin=0, ymax=1)
    ax.text(50, 1.05, 'Interface\n(x=50)', ha='center', va='bottom', fontsize=14, fontweight='bold')

    # ==============================
    # 4. 区域文字标注
    # ==============================
    ax.text(25, 0.5, 'Region 1', 
            ha='center', va='center', fontsize=16, fontweight='bold', color='darkblue')
    
    ax.text(75, 0.5, 'Region 2', 
            ha='center', va='center', fontsize=16, fontweight='bold', color='darkred')

    # ==============================
    # 5. 装饰与显示
    # ==============================
    
    # 设置显示范围
    ax.set_xlim(-25, 125)
    ax.set_ylim(-0.8, 1.4)
    
    # 隐藏Y轴刻度 (因为是1D问题，Y轴无意义)
    ax.axes.get_yaxis().set_visible(False)
    # 隐藏边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # 只保留底部的轴线用于看坐标
    ax.spines['bottom'].set_position(('data', 0))
    
    # 标题
    ax.set_title('1D Slab Reactor Model (Monte Carlo)', fontsize=20, pad=20)
    ax.set_xlabel('x (cm)', fontsize=16)
    
    # 刻度字体
    plt.xticks(fontsize=14)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    draw_mc_slab_geometry()