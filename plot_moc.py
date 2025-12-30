import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_pin_cell_geometry():
    # ==============================
    # 1. 几何参数 (来自你的 MOC 代码)
    # ==============================
    pitch = 1.26          # 栅距 (cm)
    radius = 0.4096       # 燃料芯块半径 (cm)
    
    # 绘图设置
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    
    # ==============================
    # 2. 绘制区域
    # ==============================
    
    # --- 区域 B: 慢化剂 (Moderator/Water) ---
    # 画一个正方形充满背景
    # 左下角坐标 (-p/2, -p/2)
    p2 = pitch / 2.0
    rect = patches.Rectangle((-p2, -p2), pitch, pitch, 
                             linewidth=2, edgecolor='black', facecolor='#A0D4FF', 
                             label='Moderator (H2O)\nLow Absorption\nHigh Scattering')
    ax.add_patch(rect)
    
    # --- 区域 A: 燃料 (Fuel/UO2) ---
    # 画一个圆在中心
    circle = patches.Circle((0, 0), radius, 
                            linewidth=1.5, edgecolor='black', facecolor='#FF6B6B', 
                            label='Fuel (UO2)\nHigh Absorption\nFission Source')
    ax.add_patch(circle)

    # ==============================
    # 3. 添加标注 (Dimensions)
    # ==============================
    
    # --- 标注栅距 (Pitch) ---
    # 在上方画双箭头线
    ax.annotate('', xy=(-p2, p2 + 0.1), xytext=(p2, p2 + 0.1),
                arrowprops=dict(arrowstyle='<->', lw=1.5))
    ax.text(0, p2 + 0.12, f'Pitch = {pitch} cm', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # --- 标注半径 (Radius) ---
    # 从圆心画一条线到边缘
    theta = np.deg2rad(45) # 45度角
    rx, ry = radius * np.cos(theta), radius * np.sin(theta)
    ax.annotate('', xy=(0, 0), xytext=(rx, ry),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.text(rx + 0.05, ry + 0.05, f'Radius = {radius} cm', fontsize=12, fontweight='bold', color='darkred')
    
    # --- 标注实际包壳 (虽然计算忽略了，但画出来更有物理感) ---
    # 画一个虚线圆表示包壳位置 (Cladding)
    # 典型的西屋 17x17 包壳外径约 0.475 cm
    clad_radius = 0.475
    clad_circle = patches.Circle((0, 0), clad_radius, 
                                 linewidth=1, edgecolor='gray', facecolor='none', linestyle='--',
                                 label='(Ignored Cladding)')
    ax.add_patch(clad_circle)
    ax.text(0.35, -0.4, 'Cladding (Ignored in Model)', color='gray', fontsize=9, style='italic')

    # ==============================
    # 4. 图表装饰
    # ==============================
    ax.set_xlim(-p2 - 0.2, p2 + 0.2)
    ax.set_ylim(-p2 - 0.2, p2 + 0.3)
    ax.set_aspect('equal')
    ax.set_title('PWR Pin Cell Geometry (Westinghouse 17x17)', fontsize=16, pad=20)
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    
    # 隐藏坐标轴刻度，只保留边框
    # ax.axis('off') 
    
    ax.legend(loc='lower right', framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    draw_pin_cell_geometry()