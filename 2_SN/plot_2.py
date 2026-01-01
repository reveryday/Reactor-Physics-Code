import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def draw_beautiful_analysis():
    # ==========================================
    # 1. 准备数据
    # ==========================================
    sn_orders = [2, 4, 8, 12]
    max_fluxes = [
        9.4958e14,  # S2
        9.2985e14,  # S4
        9.6992e14,  # S8
        9.7760e14   # S12 (基准)
    ]
    
    # 计算相对于 S12 的误差 (结果是 float 类型)
    ref_val = max_fluxes[-1] # S12
    errors_pct = [(val - ref_val) / ref_val * 100.0 for val in max_fluxes]
    abs_errors = [abs(e) for e in errors_pct]

    # 创建 Pandas DataFrame 用于表格展示 (这里才转换成带%的字符串)
    df = pd.DataFrame({
        'SN': [f'S{n}' for n in sn_orders],
        'Max Flux': [f'{x:.4e}' for x in max_fluxes],
        'Relative Error (%)': [f'{x:+.2f}%' for x in errors_pct]
    })

    # ==========================================
    # 2. 设置绘图风格
    # ==========================================
    # 尝试使用 seaborn 风格，如果没有安装 seaborn，回退到默认
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('ggplot') # 备用风格
    
    # 创建画布
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # 颜色定义
    line_color = '#1f77b4'  # 蓝色
    point_color = '#d62728' # 红色
    ref_color = '#2ca02c'   # 绿色

    # ==========================================
    # 3. 绘制折线图 (ax1)
    # ==========================================
    # 绘制连线
    ax1.plot(sn_orders, abs_errors, color=line_color, linestyle='--', linewidth=2, alpha=0.7, zorder=1)
    
    # 绘制数据点
    ax1.scatter(sn_orders, abs_errors, color=point_color, s=100, zorder=2, edgecolors='white', linewidth=1.5)
    
    # 特别标记 S12 (基准点)
    ax1.scatter([12], [0], color=ref_color, s=150, marker='*', zorder=3, label='Reference ($S_{12}$)')

    # 添加数值标签
    # 【修复点】：这里 val 是 float，直接使用 abs(val)
    for i, val in enumerate(errors_pct):
        y_pos = abs_errors[i]
        label_txt = f"{abs(val):.2f}%"
        
        ax1.annotate(label_txt, 
                     (sn_orders[i], y_pos), 
                     textcoords="offset points", 
                     xytext=(0, 10), 
                     ha='center', 
                     fontsize=11, 
                     fontweight='bold',
                     color='#333333',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#dddddd", alpha=0.8))

    # 图表装饰
    ax1.set_title('Convergence of $S_N$ Method (Error relative to $S_{12}$)', fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel('Absolute Relative Error (%)', fontsize=12)
    ax1.set_xlabel('$S_N$ Order', fontsize=12)
    ax1.set_xticks(sn_orders)
    ax1.set_ylim(bottom=-0.5, top=max(abs_errors) * 1.2) 
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax1.legend(loc='upper right', frameon=True, framealpha=0.9, shadow=True)

    # ==========================================
    # 4. 绘制表格 (ax2)
    # ==========================================
    ax2.axis('off')
    
    table_data = df.values
    col_labels = df.columns
    
    the_table = ax2.table(cellText=table_data,
                          colLabels=col_labels,
                          loc='center',
                          cellLoc='center',
                          colColours=['#e6f2ff']*3)

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(11)
    the_table.scale(1, 1.8) 
    
    for (row, col), cell in the_table.get_celld().items():
        cell.set_edgecolor('#dddddd')
        if row == 0:
            cell.set_text_props(weight='bold', color='#333333')
            cell.set_linewidth(1.5)
            cell.set_edgecolor('#aaaaaa')
        elif row == 4: # S12 行高亮
             cell.set_facecolor('#eaffea')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    draw_beautiful_analysis()