import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_slab_geometry():
    # 创建画布
    fig, ax = plt.subplots(figsize=(12, 6))  # 稍微调大画布尺寸以容纳大字体
    
    # 定义几何尺寸
    x_total = 100.0  # 总厚度 100 cm
    x_mid = 50.0     # 分界面 50 cm
    y_height = 1.0   # 示意图的高度（任意单位）

    # ------------------- 绘制区域 1 (0 < x < 50) -------------------
    # 参数
    # Sigma_a = 0.12, Sigma_s = 0.05, vSigma_f = 0.15
    rect1 = patches.Rectangle((0, 0), x_mid, y_height, 
                              linewidth=2, edgecolor='black', facecolor='#A9D6E5', alpha=0.8)
    ax.add_patch(rect1)
    
    # 添加区域 1 的文字标签 (字体改大至 18)
    text_region1 = (
        r"$\bf{Region\ 1}$" + "\n" +
        r"$0 < x < 50$ cm" + "\n\n" +
        r"$\Sigma_a = 0.12$ cm$^{-1}$" + "\n" +
        r"$\Sigma_s = 0.05$ cm$^{-1}$" + "\n" +
        r"$\nu\Sigma_f = 0.15$ cm$^{-1}$"
    )
    ax.text(x_mid/2, y_height/2, text_region1, ha='center', va='center', fontsize=18,
            bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.5'))

    # ------------------- 绘制区域 2 (50 < x < 100) -------------------
    # 参数
    # Sigma_a = 0.10, Sigma_s = 0.05, vSigma_f = 0.12
    rect2 = patches.Rectangle((x_mid, 0), x_total - x_mid, y_height, 
                              linewidth=2, edgecolor='black', facecolor='#F4A261', alpha=0.8)
    ax.add_patch(rect2)
    
    # 添加区域 2 的文字标签 (字体改大至 18)
    text_region2 = (
        r"$\bf{Region\ 2}$" + "\n" +
        r"$50 < x < 100$ cm" + "\n\n" +
        r"$\Sigma_a = 0.10$ cm$^{-1}$" + "\n" +
        r"$\Sigma_s = 0.05$ cm$^{-1}$" + "\n" +
        r"$\nu\Sigma_f = 0.12$ cm$^{-1}$"
    )
    ax.text(x_mid + (x_total-x_mid)/2, y_height/2, text_region2, ha='center', va='center', fontsize=18,
            bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.5'))

    # ------------------- 设置坐标轴和标注 -------------------
    ax.set_xlim(-5, 105)
    ax.set_ylim(0, 1.2)
    
    # X轴标签 (字体改大至 20)
    ax.set_xlabel("$x$ (cm)", fontsize=20)
    
    # 设置坐标轴刻度数字大小 (新增设置，大小为 16)
    ax.tick_params(axis='x', labelsize=16)
    
    # 隐藏Y轴刻度（因为是一维几何，Y方向无物理意义）
    ax.set_yticks([])
    
    # 添加底部的刻度线标注
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=50, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=100, color='black', linestyle='--', alpha=0.5)
    
    # 标题 (字体改大至 24)
    ax.set_title("1D Slab Reactor Geometry Setup (One-Speed Neutrons)", fontsize=24, pad=20)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    draw_slab_geometry()