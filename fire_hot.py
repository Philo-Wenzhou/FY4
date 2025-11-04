import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib
from scipy.ndimage import generic_filter
from numpy import deg2rad, tan, arctan, sqrt, cos, sin

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# =================== 1. 常量定义 ====================
# 地球参数 (FY-4B全局属性)
EARTH_SEMI_MAJOR = 6378137.0     # 地球长半轴 [m]
EARTH_SEMI_MINOR = 6356752.31414  # 地球短半轴 [m]
SAT_HEIGHT = 42164000             # 卫星高度 [m] (静止轨道)
SUB_LON = 105.0                   # 星下点经度 [°] (FY-4B位置)

# 火点检测阈值 (论文公式)
T7_ABSOLUTE_THRESH = 300          # 3.75μm通道绝对温度阈值 [K]
VIS_REFL_THRESH = 0.7             # 可见光反射率阈值
SUN_ZENITH_THRESH = 3             # 最小太阳天顶角 [°]
BTD_THRESH_BASE = 5               # 基础亮温差阈值 [K]
REFL_ADJUST_FACTOR = 100          # 反射率调整因子

# ===== 中国区域网格参数 ===== 
LAT_MIN = 18.0   # 最小纬度 (y_min)
LAT_MAX = 54.0   # 最大纬度 (y_max)
LON_MIN = 72.0   # 最小经度 (x_min)
LON_MAX = 136.0  # 最大经度 (x_max)
GRID_STEP = 0.04 # 网格步长

# 数据目录
DATA_DIR = 'E:\\FY4BPro\\data\\'

# =================== 2. 辅助函数 ====================
def dn_to_reflectance(dn, scale, offset):
    """DN值转反射率 (公式1)"""
    return dn * scale + offset

def dn_to_radiance(dn, scale, offset):
    """DN值转辐射亮度 (热红外通道)"""
    return dn * scale + offset

def radiance_to_bt(radiance, wavelength):
    """辐射亮度转亮温 (普朗克反演)"""
    # 添加小量防止log(0)错误
    radiance = np.maximum(radiance, 1e-10)
    
    c1 = 1.19104e8   # W/(m²·sr·μm⁴)
    c2 = 14387.8     # μm·K
    
    # 避免除以零和无效值
    with np.errstate(divide='ignore', invalid='ignore'):
        bt = c2 / (wavelength * np.log(c1 / (wavelength**5 * radiance) + 1))
    
    return np.where(np.isfinite(bt), bt, np.nan)

def adjust_reflectance(refl, sza):
    """反射率太阳天顶角校正 (公式2)"""
    # 过滤无效太阳天顶角
    valid_sza = np.where(sza >= 0, sza, np.nan)
    sza_rad = deg2rad(valid_sza)
    
    # 安全计算调整因子
    with np.errstate(invalid='ignore'):
        adjustment = np.cos(sza_rad * (1.0 - 1.3 * np.sin(0.05 * sza_rad)))
    
    # 避免除以零
    adjustment = np.where(adjustment > 0.1, adjustment, 0.1)
    
    return refl / adjustment

def compute_geocentric_lat(lat):
    """计算地心纬度 (公式4)"""
    lat_rad = deg2rad(lat)
    return np.arctan((EARTH_SEMI_MINOR**2 / EARTH_SEMI_MAJOR**2) * np.tan(lat_rad))

def compute_projection_distance(phi_e):
    """计算投影距离 (公式5)"""
    ecc_sq = (EARTH_SEMI_MAJOR**2 - EARTH_SEMI_MINOR**2) / EARTH_SEMI_MAJOR**2
    return EARTH_SEMI_MINOR / np.sqrt(1 - ecc_sq * np.cos(phi_e)**2)

def read_geo_data(geo_file, data_shape=None):
    """读取GEO文件中的经纬度和太阳天顶角数据
    
    参数:
        geo_file: GEO文件路径
        data_shape: L1数据的形状(rows, cols)，用于创建匹配的默认网格
    
    返回:
        经纬度和太阳天顶角数据数组
    """
    print(f"正在读取GEO文件: {geo_file}")
    with h5py.File(geo_file, 'r') as f_geo:
        # 打印文件结构以帮助调试
        print("GEO文件结构:")
        for key in f_geo.keys():
            print(f"  - 组: {key}")
        
        # 检查Navigation组中的数据
        if 'Navigation' in f_geo:
            print("  Navigation组包含:")
            for nav_key in f_geo['Navigation'].keys():
                print(f"    * {nav_key}")
        
        # 尝试从Navigation组读取数据
        lat_geo = None
        lon_geo = None
        sun_zenith = None
        
        # 查找经纬度数据
        if 'Navigation' in f_geo:
            for key in f_geo['Navigation'].keys():
                if 'Latitude' in key:
                    lat_geo = f_geo[f'Navigation/{key}'][:]
                    print(f"找到纬度数据: Navigation/{key}")
                if 'Longitude' in key:
                    lon_geo = f_geo[f'Navigation/{key}'][:]
                    print(f"找到经度数据: Navigation/{key}")
                if 'SunZenith' in key or 'Sun_Zenith' in key:
                    sun_zenith = f_geo[f'Navigation/{key}'][:]
                    print(f"找到太阳天顶角数据: Navigation/{key}")
        
        # 如果仍然找不到数据，尝试其他路径
        if lat_geo is None or lon_geo is None:
            # 尝试直接在根目录查找
            for key in f_geo.keys():
                if 'Latitude' in key and lat_geo is None:
                    lat_geo = f_geo[key][:]
                    print(f"找到纬度数据: {key}")
                if 'Longitude' in key and lon_geo is None:
                    lon_geo = f_geo[key][:]
                    print(f"找到经度数据: {key}")
        
        # 如果还是找不到，创建默认网格
        if lat_geo is None or lon_geo is None:
            print("警告: 未找到经纬度数据，创建默认网格")
            # 优先使用L1数据的形状，如果没有提供则使用默认值
            if data_shape and len(data_shape) == 2:
                rows, cols = data_shape
                print(f"根据L1数据形状创建网格: {rows}x{cols}")
            else:
                rows, cols = 2748, 2748  # 使用与L1数据匹配的默认值
                print(f"使用默认网格大小: {rows}x{cols}")
            
            # 创建覆盖中国区域的经纬度网格
            lat_geo = np.linspace(LAT_MAX, LAT_MIN, rows).reshape(rows, 1) * np.ones((1, cols))
            lon_geo = np.linspace(LON_MIN, LON_MAX, cols).reshape(1, cols) * np.ones((rows, 1))
        
        # 如果找不到太阳天顶角数据，创建默认值
        if sun_zenith is None:
            print("警告: 未找到太阳天顶角数据，使用默认值")
            sun_zenith = np.zeros_like(lat_geo)
        
        # 数据质量控制
        lat_geo = np.where((lat_geo >= -90) & (lat_geo <= 90), lat_geo, np.nan)
        lon_geo = np.where((lon_geo >= -180) & (lon_geo <= 180), lon_geo, np.nan)
        sun_zenith = np.where((sun_zenith >= 0) & (sun_zenith <= 180), sun_zenith, np.nan)
        
    return lat_geo, lon_geo, sun_zenith

# =================== 3. 主处理流程 ====================
def detect_hotspots(geo_file, data_file, output_csv, output_plot):
    """主函数：检测火点并输出结果"""
    print(f"开始处理FY-4B数据...")
    
    # 先读取L1数据以获取正确的数据形状
    with h5py.File(data_file, 'r') as f:
        # 获取第一个通道的数据形状作为参考
        data_shape = None
        if 'Data' in f:
            for key in f['Data'].keys():
                data_shape = f[f'Data/{key}'].shape
                print(f"L1数据形状: {data_shape}")
                break
    
    # =================== 3.1 读取GEO数据 ====================
    lat, lon, sun_zenith = read_geo_data(geo_file, data_shape)
    print(f"GEO数据形状: {lat.shape}")
    print(f"使用WGS84 (EPSG:4326)坐标系")
    
    # ===== 使用指定网格参数生成中国区域网格 =====
    print(f"使用指定网格参数: 纬度{LAT_MIN}°-{LAT_MAX}°, 经度{LON_MIN}°-{LON_MAX}°, 步长{GRID_STEP}")
    
    # 获取中国区域的索引（直接剪裁）
    region_mask = (
        (lat >= LAT_MIN) & (lat <= LAT_MAX) &
        (lon >= LON_MIN) & (lon <= LON_MAX)
    )
    
    # 获取中国区域的边界索引
    y_indices, x_indices = np.where(region_mask)
    if len(y_indices) > 0:
        y_min_crop, y_max_crop = np.min(y_indices), np.max(y_indices)
        x_min_crop, x_max_crop = np.min(x_indices), np.max(x_indices)
        print(f"中国区域边界索引: y[{y_min_crop}:{y_max_crop}], x[{x_min_crop}:{x_max_crop}]")
    else:
        print("警告: 在中国区域范围内未找到数据点")
        y_min_crop, y_max_crop, x_min_crop, x_max_crop = 0, lat.shape[0]-1, 0, lat.shape[1]-1
    
    # =================== 3.2 读取L1数据并定标 ====================
    print(f"正在读取L1数据文件: {data_file}")
    with h5py.File(data_file, 'r') as f:
        # 打印文件结构
        print("L1文件结构:")
        for key in f.keys():
            print(f"  - 组: {key}")
        
        if 'Data' in f:
            print("  Data组包含:")
            for data_key in f['Data'].keys():
                print(f"    * {data_key}")
        
        # 尝试找到可见光通道
        vis_dn = None
        if 'Data' in f:
            for key in f['Data'].keys():
                if 'Channel02' in key or 'VIS' in key:
                    vis_dn = f[f'Data/{key}'][:]
                    print(f"找到可见光通道: Data/{key}")
                    break
        
        # 尝试找到红外通道
        ch07_dn = None
        ch13_dn = None
        if 'Data' in f:
            for key in f['Data'].keys():
                if ('Channel07' in key or '3.7' in key) and ch07_dn is None:
                    ch07_dn = f[f'Data/{key}'][:]
                    print(f"找到3.75μm通道: Data/{key}")
                if ('Channel13' in key or '10.8' in key) and ch13_dn is None:
                    ch13_dn = f[f'Data/{key}'][:]
                    print(f"找到10.8μm通道: Data/{key}")
        
        # 获取定标系数
        vis_scale, vis_offset = 1.0, 0.0
        ch07_scale, ch07_offset = 1.0, 0.0
        ch13_scale, ch13_offset = 1.0, 0.0
        
        if 'Calibration' in f:
            print("  Calibration组包含:")
            for calib_key in f['Calibration'].keys():
                print(f"    * {calib_key}")
            
            # 尝试获取定标系数
            if 'CALIBRATION_COEF(SCALE+OFFSET)' in f['Calibration']:
                calib_coef = f['Calibration']['CALIBRATION_COEF(SCALE+OFFSET)'][:]
                print(f"定标系数形状: {calib_coef.shape}")
                
                # 假设通道索引是固定的
                if calib_coef.shape[0] >= 13:
                    vis_scale = calib_coef[1, 0]  # 索引1对应通道2
                    vis_offset = calib_coef[1, 1]
                    ch07_scale = calib_coef[6, 0]  # 索引6对应通道7
                    ch07_offset = calib_coef[6, 1]
                    ch13_scale = calib_coef[12, 0]  # 索引12对应通道13
                    ch13_offset = calib_coef[12, 1]
    
    # 确保所有数据都被正确读取
    if vis_dn is None:
        print("警告: 未找到可见光通道数据，创建默认数据")
        vis_dn = np.zeros_like(lat)
    if ch07_dn is None:
        print("警告: 未找到3.75μm通道数据，创建默认数据")
        ch07_dn = np.zeros_like(lat)
    if ch13_dn is None:
        print("警告: 未找到10.8μm通道数据，创建默认数据")
        ch13_dn = np.zeros_like(lat)
    
    # 验证所有数据形状是否匹配
    data_shapes = [lat.shape, lon.shape, sun_zenith.shape, vis_dn.shape, ch07_dn.shape, ch13_dn.shape]
    if len(set(data_shapes)) > 1:
        print("警告: 数据形状不匹配，将使用L1数据形状重新调整GEO数据")
        target_shape = vis_dn.shape  # 使用L1数据形状作为目标
        
        # 重新创建匹配形状的经纬度网格
        rows, cols = target_shape
        lat = np.linspace(LAT_MAX, LAT_MIN, rows).reshape(rows, 1) * np.ones((1, cols))
        lon = np.linspace(LON_MIN, LON_MAX, cols).reshape(1, cols) * np.ones((rows, 1))
        sun_zenith = np.zeros(target_shape)
        
        # 重新计算区域掩码
        region_mask = ((lat >= LAT_MIN) & (lat <= LAT_MAX) & (lon >= LON_MIN) & (lon <= LON_MAX))
        y_indices, x_indices = np.where(region_mask)
        if len(y_indices) > 0:
            y_min_crop, y_max_crop = np.min(y_indices), np.max(y_indices)
            x_min_crop, x_max_crop = np.min(x_indices), np.max(x_indices)
            print(f"更新后的中国区域边界索引: y[{y_min_crop}:{y_max_crop}], x[{x_min_crop}:{x_max_crop}]")
    
    # =================== 3.4 先剪裁中国区域数据，再进行后续计算 ====================
    # 剪裁所有数据到中国区域 - 提高计算效率
    print(f"正在剪裁中国区域数据 (EPSG:4326)...")
    print("先剪裁再计算，避免对全球数据进行不必要的处理")
    
    # 剪裁地理数据
    lat_cropped = lat[y_min_crop:y_max_crop+1, x_min_crop:x_max_crop+1]
    lon_cropped = lon[y_min_crop:y_max_crop+1, x_min_crop:x_max_crop+1]
    sun_zenith_cropped = sun_zenith[y_min_crop:y_max_crop+1, x_min_crop:x_max_crop+1]
    
    # 剪裁原始数据
    vis_dn_cropped = vis_dn[y_min_crop:y_max_crop+1, x_min_crop:x_max_crop+1]
    ch07_dn_cropped = ch07_dn[y_min_crop:y_max_crop+1, x_min_crop:x_max_crop+1]
    ch13_dn_cropped = ch13_dn[y_min_crop:y_max_crop+1, x_min_crop:x_max_crop+1]
    
    print(f"剪裁后的数据形状: {lat_cropped.shape}")
    
    # 定标转换 - 只处理中国区域数据
    print("正在进行数据定标...")
    vis_refl_cropped = dn_to_reflectance(vis_dn_cropped, vis_scale, vis_offset)
    vis_refl_adj_cropped = adjust_reflectance(vis_refl_cropped, sun_zenith_cropped)  # 太阳天顶角校正
    
    # 计算亮温
    ch07_rad_cropped = dn_to_radiance(ch07_dn_cropped, ch07_scale, ch07_offset)
    ch13_rad_cropped = dn_to_radiance(ch13_dn_cropped, ch13_scale, ch13_offset)
    bt_07_cropped = radiance_to_bt(ch07_rad_cropped, 3.75)  # 3.75μm亮温
    bt_13_cropped = radiance_to_bt(ch13_rad_cropped, 10.8)  # 10.8μm亮温
    
    # 计算亮温差
    btd_cropped = bt_07_cropped - bt_13_cropped
    
    # =================== 3.3 火点检测算法 - 仅在中国区域进行 ====================
    print("正在进行火点检测 (仅在中国区域范围内)...")
    
    # 基础检测条件
    hotspots = (
        (bt_07_cropped > T7_ABSOLUTE_THRESH) &  # 3.75μm通道绝对温度阈值
        (sun_zenith_cropped > SUN_ZENITH_THRESH)  # 太阳天顶角条件
    )
    
    # 昼夜分别处理
    # 白天：考虑可见光反射率和亮温差
    daytime = sun_zenith_cropped < 90
    hotspots_day = hotspots & daytime & \
                   (vis_refl_adj_cropped < VIS_REFL_THRESH) & \
                   (btd_cropped > BTD_THRESH_BASE)
    
    # 夜间：主要依赖亮温和亮温差
    nighttime = ~daytime
    hotspots_night = hotspots & nighttime & \
                      (bt_07_cropped > T7_ABSOLUTE_THRESH + 20)  # 夜间阈值更高
    
    # 合并昼夜检测结果
    hotspots_final_cropped = hotspots_day | hotspots_night
    
    # 计算火点数量
    hotspots_count = np.sum(hotspots_final_cropped)
    print(f"检测到的火点数量: {hotspots_count}")
    
    # =================== 3.4 输出结果 ====================
    # 保存火点CSV文件
    if hotspots_count > 0:
        y_coords, x_coords = np.where(hotspots_final_cropped)
        # 注意这里使用剪裁后的数据
        hotspot_lats = lat_cropped[y_coords, x_coords]
        hotspot_lons = lon_cropped[y_coords, x_coords]
        hotspot_bt07 = bt_07_cropped[y_coords, x_coords]
        hotspot_bt13 = bt_13_cropped[y_coords, x_coords]
        hotspot_btd = btd_cropped[y_coords, x_coords]
        
        # 创建DataFrame
        hotspot_df = pd.DataFrame({
            'Latitude': hotspot_lats,
            'Longitude': hotspot_lons,
            'BT_3.75um': hotspot_bt07,
            'BT_10.8um': hotspot_bt13,
            'BTD': hotspot_btd
        })
        
        # ===== 根据指定网格参数生成标准网格并聚合火点 =====
        print(f"根据网格步长 {GRID_STEP}° 生成标准网格并聚合火点...")
        
        # 生成标准网格
        lat_grid = np.arange(LAT_MIN, LAT_MAX + GRID_STEP, GRID_STEP)
        lon_grid = np.arange(LON_MIN, LON_MAX + GRID_STEP, GRID_STEP)
        
        # 为每个火点找到对应的网格索引
        lat_indices = np.searchsorted(lat_grid, hotspot_df['Latitude']) - 1
        lon_indices = np.searchsorted(lon_grid, hotspot_df['Longitude']) - 1
        
        # 过滤掉超出网格范围的点
        valid_mask = (lat_indices >= 0) & (lat_indices < len(lat_grid) - 1) & \
                     (lon_indices >= 0) & (lon_indices < len(lon_grid) - 1)
        
        if np.any(valid_mask):
            # 创建临时DataFrame存储有效火点
            valid_df = hotspot_df[valid_mask].copy()
            # 为有效火点分配网格中心坐标
            valid_df['Grid_Latitude'] = lat_grid[lat_indices[valid_mask]] + GRID_STEP / 2
            valid_df['Grid_Longitude'] = lon_grid[lon_indices[valid_mask]] + GRID_STEP / 2
            
            # 按网格聚合火点数据
            grid_agg = valid_df.groupby(['Grid_Latitude', 'Grid_Longitude']).agg({
                'Latitude': 'mean',
                'Longitude': 'mean',
                'BT_3.75um': 'mean',
                'BT_10.8um': 'mean',
                'BTD': 'mean',
            }).reset_index()
            
            # 添加网格内火点数量统计
            grid_agg['Hotspot_Count'] = valid_df.groupby(['Grid_Latitude', 'Grid_Longitude']).size().values
            
            print(f"聚合后网格火点数量: {len(grid_agg)}")
            
            # 保存聚合结果到单独的CSV文件
            grid_csv = output_csv.replace('.csv', '_grid.csv')
            grid_agg.to_csv(grid_csv, index=False)
            print(f"网格聚合火点数据已保存至: {grid_csv}")
        
        # 保存原始火点数据
        hotspot_df.to_csv(output_csv, index=False)
        print(f"火点数据已保存至: {output_csv}")
    else:
        # 创建空的CSV文件
        pd.DataFrame(columns=['Latitude', 'Longitude', 'BT_3.75um', 'BT_10.8um', 'BTD']).to_csv(output_csv, index=False)
        print(f"未检测到火点，已创建空结果文件: {output_csv}")
    
    # 创建预览图 - 使用剪裁后的数据
    plt.figure(figsize=(12, 8))
    
    # 显示3.75μm亮温
    plt.subplot(221)
    plt.imshow(bt_07_cropped, cmap='jet', vmin=280, vmax=330)
    plt.colorbar(label='亮温 (K)')
    plt.title('3.75μm 亮温 (EPSG:4326)')
    
    # 显示亮温差
    plt.subplot(222)
    plt.imshow(btd_cropped, cmap='bwr', vmin=-10, vmax=10)
    plt.colorbar(label='亮温差 (K)')
    plt.title('亮温差 (3.75μm - 10.8μm)')
    
    # 显示火点检测结果
    plt.subplot(223)
    plt.imshow(np.where(hotspots_final_cropped, 1, 0), cmap='RdYlGn_r')
    plt.colorbar(label='火点')
    plt.title(f'火点检测结果 (总数: {hotspots_count})')
    
    # 显示剪裁后的中国区域范围
    plt.subplot(224)
    # 创建简单的经纬度网格用于显示
    plt.imshow(np.ones_like(lat_cropped), cmap='Greys')
    # 添加经纬度标记和网格参数
    plt.text(10, 10, f"经度范围: {lon_cropped.min():.2f}°E - {lon_cropped.max():.2f}°E", color='white', fontsize=10, backgroundcolor='black')
    plt.text(10, 30, f"纬度范围: {lat_cropped.min():.2f}°N - {lat_cropped.max():.2f}°N", color='white', fontsize=10, backgroundcolor='black')
    plt.text(10, 50, f"网格步长: {GRID_STEP}°", color='white', fontsize=10, backgroundcolor='black')
    plt.text(10, 70, f"坐标系: EPSG:4326 (WGS84)", color='white', fontsize=10, backgroundcolor='black')
    plt.title('中国区域范围 (剪裁后)')
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"结果图像已保存至: {output_plot}")
    
    return hotspots_count

# =================== 4. 主函数 ====================
def main():
    print("========== FY-4B 异常热点检测系统 ==========")
    
    # 设置文件路径 - 使用测试脚本找到的正确文件
    geo_file = os.path.join(DATA_DIR, 'Z_SATE_C_BAWX_20250904011635_P_FY4B-_AGRI--_N_DISK_1050E_L1-_GEO-_MULT_NOM_20250904010000_20250904011459_4000M_V0001.HDF')
    data_file = os.path.join(DATA_DIR, 'Z_SATE_C_BAWX_20250904013138_P_FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20250904011500_20250904012959_4000M_V0001.HDF')
    
    # 创建输出目录
    output_dir = 'e:\\FY4BPro\\fire\\output_hotspots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 设置输出文件路径
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    output_csv = os.path.join(output_dir, f'hotspots_{timestamp}.csv')
    output_plot = os.path.join(output_dir, f'hotspots_detection_{timestamp}.png')
    
    # 检查输入文件是否存在
    if not os.path.exists(geo_file):
        print(f"错误: GEO文件不存在: {geo_file}")
        return
    if not os.path.exists(data_file):
        print(f"错误: L1数据文件不存在: {data_file}")
        return
    
    # 执行火点检测
    try:
        hotspots_count = detect_hotspots(geo_file, data_file, output_csv, output_plot)
        print(f"\n检测完成！共检测到 {hotspots_count} 个异常热点。")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

