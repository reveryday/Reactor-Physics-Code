import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_reactor_geometry_2d():
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    
    # ==============================
    # 1. 绘制区域 (使用矩形 Patch)
    # ==============================
    
    # --- Reflector (反射层) ---
    # 先画一个大的背景矩形代表整个反应堆，包括反射层
    # 范围 x[-65, 65], y[0, 120]
    # 宽度 = 130, 高度 = 120
    rect_refl = patches.Rectangle((-65, 0), 130, 120, 
                                  linewidth=2, edgecolor='black', facecolor='#E0E0E0', 
                                  label='Reflector')
    ax.add_patch(rect_refl)
    
    # --- Core 1 (活性区下部) ---
    # 范围 x[-50, 50], y[15, 55]
    # 宽度 = 100, 高度 = 40 (55-15)
    # 起点 (-50, 15)
    rect_c1 = patches.Rectangle((-50, 15), 100, 40, 
                                linewidth=1.5, edgecolor='black', facecolor='#FF9999', 
                                label='Core Region 1')
    ax.add_patch(rect_c1)
    
    # --- Core 2 (活性区上部) ---
    # 范围 x[-50, 50], y[55, 105]
    # 宽度 = 100, 高度 = 50 (105-55)
    # 起点 (-50, 55)
    rect_c2 = patches.Rectangle((-50, 55), 100, 50, 
                                linewidth=1.5, edgecolor='black', facecolor='#FFCC99', 
                                label='Core Region 2')
    ax.add_patch(rect_c2)

    # ==============================
    # 2. 标注尺寸与辅助线
    # ==============================
    
    # 辅助函数：画双箭头标注
    # 修改：将默认字体从 10 增大到 14
    def add_dim_arrow(x1, y1, x2, y2, text, offset_text=0):
        ax.annotate('', xy=(x1, y1), xytext=(x2, y2),
                    arrowprops=dict(arrowstyle='<->', lw=1.5))
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        if x1 == x2: # 垂直标注
            ax.text(mid_x + offset_text, mid_y, text, rotation=90, va='center', ha='left', fontsize=14)
        else: # 水平标注
            ax.text(mid_x, mid_y + offset_text, text, va='bottom', ha='center', fontsize=14)

    # --- Y轴尺寸标注 (画在右侧) ---
    # 底部反射层厚度 (0-15)
    add_dim_arrow(75, 0, 75, 15, '15cm', 2)
    # Core 1 高度 (15-55)
    add_dim_arrow(75, 15, 75, 55, '40cm', 2)
    # Core 2 高度 (55-105)
    add_dim_arrow(75, 55, 75, 105, '50cm', 2)
    # 顶部反射层厚度 (105-120)
    add_dim_arrow(75, 105, 75, 120, '15cm', 2)
    
    # --- X轴尺寸标注 (画在底部) ---
    # Core 宽度 (-50 到 50)
    add_dim_arrow(-50, -10, 50, -10, 'Core Width: 100cm', 2)
    # 左反射层 (-65 到 -50)
    add_dim_arrow(-65, -10, -50, -10, '15cm', 2)
    
    # --- 关键坐标点虚线 ---
    # X = -50, 50
    ax.vlines([-50, 50], 0, 120, colors='black', linestyles=':', alpha=0.5)
    # Y = 15, 55, 105
    ax.hlines([15, 55, 105], -65, 65, colors='black', linestyles=':', alpha=0.5)

    # ==============================
    # 3. 装饰与显示
    # ==============================
    
    # 设置坐标轴范围 (留出空白写字)
    ax.set_xlim(-80, 100)
    ax.set_ylim(-20, 130)
    ax.set_aspect('equal')
    
    # 修改：增大标题字体到 20
    ax.set_title('Reactor Geometry Model', fontsize=20, pad=15)
    # 修改：增大轴标签字体到 16
    ax.set_xlabel('X(cm)', fontsize=16)
    ax.set_ylabel('Y(cm)', fontsize=16)
    
    # 修改：增大刻度字体到 14
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # 文本标注区域中心
    # 修改：增大区域文字字体到 16，反射层到 12
    ax.text(0, 35, 'Region 1', ha='center', va='center', fontweight='bold', color='darkred', fontsize=16)
    ax.text(0, 80, 'Region 2', ha='center', va='center', fontweight='bold', color='darkorange', fontsize=16)
    ax.text(0, 7.5, 'Reflector', ha='center', va='center', fontsize=12, color='dimgray')
    ax.text(0, 112.5, 'Reflector', ha='center', va='center', fontsize=12, color='dimgray')
    
    # 添加图例
    # 修改：增大图例字体到 13
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=13)
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    draw_reactor_geometry_2d()