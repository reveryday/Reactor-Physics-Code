import matplotlib.pyplot as plt
import numpy as np

def plot_sn_all_errors():
    # ==============================
    # 1. 数据准备
    # ==============================
    n_values = np.array([2, 4, 8, 16, 32])
    
    # 整理数据：每一行代表一个N，每一列代表一个位置 (0.25, 0.50, 0.75, 1.00)
    # 数据来源：你提供的运行日志
    errors_data = {
        'x/a = 0.25': [0.2932, 0.0551, 0.0118, 0.0032, 0.0013],
        'x/a = 0.50': [1.2666, 0.2883, 0.0471, 0.0130, 0.0042],
        'x/a = 0.75': [3.1625, 1.2431, 0.1091, 0.0373, 0.0106],
        'x/a = 1.00': [6.5893, 0.4192, 0.0840, 0.0181, 0.0030]
    }
    
    # 定义样式：颜色和标记
    styles = {
        'x/a = 0.25': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},  # 蓝
        'x/a = 0.50': {'color': '#2ca02c', 'marker': 's', 'linestyle': '-'},  # 绿
        'x/a = 0.75': {'color': '#ff7f0e', 'marker': '^', 'linestyle': '-'},  # 橙
        'x/a = 1.00': {'color': '#d62728', 'marker': '*', 'linestyle': '--'}  # 红 (边界)
    }

    # ==============================
    # 2. 绘图设置
    # ==============================
    fig, ax = plt.subplots(figsize=(10, 7), dpi=120) # 稍微大一点，PPT更清晰
    
    # 3. 循环绘制四条线
    for label, data in errors_data.items():
        style = styles[label]
        ax.plot(n_values, data, label=label, 
                color=style['color'], 
                marker=style['marker'], 
                linestyle=style['linestyle'],
                linewidth=2, markersize=8)
        
        # 标注 S32 的最终精度值
        final_val = data[-1]
        ax.annotate(f'{final_val:.4f}%', 
                    (32, final_val), 
                    textcoords="offset points", 
                    xytext=(5, 0), 
                    ha='left', va='center', fontsize=10, color=style['color'], fontweight='bold')

    # 4. 坐标轴设置 (关键：对数坐标)
    ax.set_yscale('log')
    
    ax.set_xticks(n_values)
    ax.set_xticklabels([f'S{n}' for n in n_values], fontsize=12, fontweight='bold')
    
    # 设置范围，留出一点空间给文字
    ax.set_xlim(1.5, 38)
    
    # 5. 标签与标题
    ax.set_xlabel(r'Angular Quadrature Order ($N$)', fontsize=14)
    ax.set_ylabel(r'Relative Error (%)', fontsize=14)
    ax.set_title(r'Convergence Analysis of $S_N$ Method', fontsize=18, fontweight='bold', pad=20)
    
    # 6. 网格与图例
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.grid(True, which="major", ls="-", alpha=0.6)
    
    ax.legend(fontsize=12, loc='upper right', frameon=True, shadow=True)
    

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_sn_all_errors()