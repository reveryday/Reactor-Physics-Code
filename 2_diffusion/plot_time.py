import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置数据
step_sizes = [0.25, 0.5, 1.0, 2.5, 5]
calc_times = [334.01, 45.23, 6.45, 0.66, 0.20]

# 创建图表
plt.figure(figsize=(10, 6))
plt.plot(step_sizes, calc_times, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8)

# 添加数值标签
for x, y in zip(step_sizes, calc_times):
    plt.text(x, y + 10, f'{y}', ha='center', va='bottom', fontsize=10)

# 设置标题和标签 (使用通用字体以避免中文乱码，或者英文标签)
plt.title('Relationship between Grid Step and Computation Time', fontsize=14)
plt.xlabel('Grid Step Size', fontsize=12)
plt.ylabel('Computation Time', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# 反转X轴 (通常步长越小越精密，放在左边或右边皆可，这里按数轴自然顺序)
# 如果想强调"加密网格"，可以保留自然顺序，显示左侧0.25处极高

# 保存图像
plt.tight_layout()
plt.savefig('grid_time_plot.png')