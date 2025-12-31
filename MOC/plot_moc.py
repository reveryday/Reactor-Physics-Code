import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_moc_geometry_and_rays(pitch, radius, angle_deg=45, n_rays=15):
    """
    绘制MOC几何及特定角度下的特征线轨迹
    """
    # 设置绘图风格
    plt.style.use('fast') 
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 1. 绘制几何实体
    # 慢化剂区域 (方形边框)
    half_p = pitch / 2.0
    rect = patches.Rectangle((-half_p, -half_p), pitch, pitch, 
                             linewidth=2, edgecolor='black', facecolor='#E0F7FA', label='Moderator')
    ax.add_patch(rect)
    
    # 燃料区域 (圆形)
    circle = patches.Circle((0, 0), radius, 
                            linewidth=2, edgecolor='darkred', facecolor='#FFCDD2', label='Fuel')
    ax.add_patch(circle)
    
    # 2. 计算并绘制特征线 (Rays)
    # 这里的逻辑是为了可视化，模拟 MOC 的射线生成
    phi = np.radians(angle_deg)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    
    # 计算投影宽度以确定射线范围
    # 投影到垂直于射线的轴上
    max_proj = half_p * (np.abs(sin_phi) + np.abs(cos_phi))
    
    # 生成一系列垂直距离 t
    t_values = np.linspace(-max_proj, max_proj, n_rays)
    
    print(f"正在绘制 {angle_deg}° 方向的 {len(t_values)} 条特征线...")
    
    for t in t_values:
        # 射线方程: x*sin(phi) - y*cos(phi) = t
        # 转化为 y = mx + c 形式方便绘图: y = x*tan(phi) - t/cos(phi)
        
        points = []
        
        # 计算射线与方形边界 x=±half_p, y=±half_p 的交点
        # x = ±half_p
        for x_edge in [-half_p, half_p]:
            if abs(cos_phi) > 1e-6:
                y = (x_edge * sin_phi - t) / cos_phi
                if -half_p - 1e-5 <= y <= half_p + 1e-5:
                    points.append([x_edge, y])
        
        # y = ±half_p
        for y_edge in [-half_p, half_p]:
            if abs(sin_phi) > 1e-6:
                x = (t + y_edge * cos_phi) / sin_phi
                if -half_p - 1e-5 <= x <= half_p + 1e-5:
                    points.append([x, y_edge])
        
        points = np.array(points)
        
        # 只有当射线穿过盒子时才绘制
        if len(points) >= 2:
            # 排序点，确保连线正确
            # 简单排序：按x坐标，如果x相同按y坐标
            points = points[np.argsort(points[:, 0])]
            p1, p2 = points[0], points[-1]
            
            # 绘制贯穿线 (细灰线)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', linestyle='-', linewidth=1, alpha=0.6)
            
            # --- 高级特效：计算与圆的交点并高亮 ---
            # 圆心到直线的距离就是 |t| (因为我们定义的t就是法向距离)
            if abs(t) < radius:
                # 弦长的一半
                half_chord = np.sqrt(radius**2 - t**2)
                
                # 圆内线段的中心点 (投影点)
                # x_c*sin + y_c*(-cos) = 0 (垂直线) & x_c*sin - y_c*cos = t
                # 解得: x_c = t*sin, y_c = -t*cos
                x_c = t * sin_phi
                y_c = -t * cos_phi
                
                # 沿射线方向的单位向量 (-cos, -sin) 或 (cos, sin)?
                # 射线方向向量是 (cos_phi, sin_phi)
                dx = cos_phi * half_chord
                dy = sin_phi * half_chord
                
                # 燃料内的线段端点
                fuel_p1 = [x_c - dx, y_c - dy]
                fuel_p2 = [x_c + dx, y_c + dy]
                
                # 绘制燃料内的轨迹段 (红色加粗)
                ax.plot([fuel_p1[0], fuel_p2[0]], [fuel_p1[1], fuel_p2[1]], 
                        color='red', linewidth=2, linestyle='-')
                
                # 绘制交点 (小圆点)
                ax.plot(fuel_p1[0], fuel_p1[1], 'o', markersize=3, color='black')
                ax.plot(fuel_p2[0], fuel_p2[1], 'o', markersize=3, color='black')

    # 3. 装饰图表
    ax.set_xlim(-half_p*1.2, half_p*1.2)
    ax.set_ylim(-half_p*1.2, half_p*1.2)
    ax.set_aspect('equal')
    ax.set_title(f'MOC Ray Tracing Visualization\n(Azimuthal Angle: {angle_deg}°)', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    
    # 添加图例
    ax.legend(loc='upper right', frameon=True, shadow=True)
    
    # 去除多余的边框刻度
    ax.minorticks_on()
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    
    # 添加一个箭头表示射线方向
    arrow_len = half_p * 0.5
    ax.arrow(-half_p, -half_p*1.1, arrow_len*cos_phi, arrow_len*sin_phi, 
             head_width=0.05, head_length=0.1, fc='blue', ec='blue', label='Direction')
    plt.text(-half_p, -half_p*1.15, "Ray Direction", color='blue', fontsize=10)

    plt.tight_layout()
    plt.show()

# ================================
# 运行配置
# ================================
# 使用与之前计算代码相同的参数
pitch_val = 1.26
radius_val = 0.41

# 这里的 n_rays 可以设大一点以显示密集感，或者设小一点看清逻辑
# 建议汇报时生成两张图：
# 1. 稀疏图 (n_rays=15)：讲解原理，展示红色的截断线段。
# 2. 密集图 (n_rays=100)：展示全域覆盖能力。
plot_moc_geometry_and_rays(pitch_val, radius_val, angle_deg=45, n_rays=20)