# -*- coding: utf-8 -*-
"""
FY-4B | 指定“某一天”的逐小时火点聚合
输入：一个目录，里面有同天的 GEO 与 FDI HDF
输出：每小时 CSV+PNG、全日统计表、GIF
依赖: h5py, numpy, pandas, matplotlib, cartopy, scipy.ndimage, imageio (或 Pillow)
"""

import os, re, sys
import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.ndimage import median_filter, label
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 可选：做 GIF
try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except Exception:
    from PIL import Image
    HAS_IMAGEIO = False

# 常量定义
BK = 9  # 中值滤波窗口大小
EPS_MAD = 1e-10  # 防止除以零的小值
HITS_PERSIST_MIN = 1  # 持久热点最少命中次数

# 中国地理范围常量
LAT_MIN = 18  # 最小纬度（南界）
LAT_MAX = 53  # 最大纬度（北界）
LON_MIN = 73  # 最小经度（西界）
LON_MAX = 135  # 最大经度（东界）

# 火点检测阈值常量（已放宽）
DAY_SZA_MAX = 75  # 白天最大太阳天顶角（度）- 放宽
NIGHT_SZA_MIN = 80  # 夜晚最小太阳天顶角（度）- 放宽
VIS_REFL_CLOUD = 0.6  # 可见光反射率云阈值 - 放宽
COLD_BT13_CLOUD = 245  # 冷云亮温阈值（K）- 放宽
ABS_T4_DAY = 310  # 白天绝对亮温阈值（K）- 放宽
ABS_T4_NGT = 290  # 夜晚绝对亮温阈值（K）- 放宽
BTD_DAY_MIN = -20  # 白天亮温差最小值 - 放宽
BTD_NGT_MIN = -10  # 夜晚亮温差最小值 - 放宽
Z_BTD_MIN = 2.5  # Z分数亮温差最小值 - 放宽
Z_T4_MIN = 2.5  # Z分数T4亮温最小值 - 放宽

def read_any(h, paths, desc="", allow_none=False):
    """尝试从多个路径读取数据，返回第一个存在的数据
    
    Args:
        h: h5py文件对象
        paths: 要尝试的数据路径列表
        desc: 数据描述，用于错误消息
        allow_none: 如果为True，则在未找到数据时返回None而不是抛出错误
    
    Returns:
        找到的数据数组或None
    """
    for p in paths:
        if p in h:
            return h[p][:]
    if allow_none:
        return None
    raise KeyError(f"{desc} not found. Tried: {paths}")

def calculate_latlon(line_num, col_num, resolution="4000M"):
    """
    根据FY-4B卫星的行列号计算经纬度
    
    参数:
        line_num: 行号数组
        col_num: 列号数组
        resolution: 分辨率，默认为"4000M"
    
    返回:
        (lat, lon) 经纬度数组
    """
    # 地球参数（与latlon2linecolumn保持一致）
    ea = 6378.137  # 地球的半长轴[km]
    eb = 6356.7523  # 地球的短半轴[km]
    h = 42164  # 地心到卫星质心的距离[km]
    λD = np.deg2rad(105)  # 卫星星下点所在经度
    
    # 分辨率参数
    COFF = {"0500M": 10991.5,
            "1000M": 5495.5,
            "2000M": 2747.5,
            "4000M": 1373.5}
    CFAC = {"0500M": 81865099,
            "1000M": 40932549,
            "2000M": 20466274,
            "4000M": 10233137}
    LOFF = COFF  # 行偏移
    LFAC = CFAC  # 行比例因子
    
    # 确保分辨率有效
    if resolution not in COFF:
        raise ValueError(f"无效的分辨率: {resolution}，支持的分辨率: {list(COFF.keys())}")
    
    # 将行列号转换为扫描角x和y（单位：度）
    x = (col_num - COFF[resolution]) * 2**16 / CFAC[resolution]
    y = (line_num - LOFF[resolution]) * 2**16 / LFAC[resolution]
    
    # 将扫描角转换为弧度
    x_rad = np.deg2rad(x)
    y_rad = np.deg2rad(y)
    
    # 计算卫星到观测点的向量分量
    r1 = -np.sin(x_rad) * np.cos(y_rad)
    r2 = -np.cos(x_rad) * np.cos(y_rad)
    r3 = -np.sin(y_rad)
    
    # 计算地球椭球面上的点
    eb2_ea2 = eb**2 / ea**2
    
    # 迭代求解地心经纬度
    phi = np.arcsin(-r3 / np.sqrt(r1**2 + r2**2 + r3**2))
    
    # 迭代计算以提高精度
    max_iter = 10
    tol = 1e-10
    
    for _ in range(max_iter):
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        
        # 计算地球椭球面上该点的曲率半径
        re = eb / np.sqrt(1 - (1 - eb2_ea2) * cos_phi**2)
        
        # 计算地心角
        sin_theta = re * np.sqrt(r1**2 + r2**2) / h
        theta = np.arcsin(np.minimum(sin_theta, 1.0))  # 确保不会超出范围
        
        # 更新phi
        phi_new = np.arcsin(-r3 * (re / h) / np.cos(theta))
        
        # 检查收敛
        if np.max(np.abs(phi_new - phi)) < tol:
            phi = phi_new
            break
        
        phi = phi_new
    
    # 计算经度
    lambda_e = λD + np.arctan2(r1, r2)
    
    # 转换为地理纬度（需要调整地心纬度到地理纬度）
    lat = np.rad2deg(np.arctan(eb2_ea2 * np.tan(phi)))
    lon = np.rad2deg(lambda_e)
    
    # 确保经度在[-180, 180]范围内
    lon = ((lon + 180.0) % 360.0) - 180.0
    
    return lat, lon

def get_calib_scale_offset(h, ch_idx):
    for p in [
        "Calibration/CALIBRATION_COEF(SCALE+OFFSET)",
        "CALIBRATION/CALIBRATION_COEF(SCALE+OFFSET)",
        "Calibration/CALIBRATION_COEF",
        "CALIBRATION/CALIBRATION_COEF",
    ]:
        if p in h:
            coef = h[p][:]
            return float(coef[ch_idx,0]), float(coef[ch_idx,1])
    scale = read_any(h, ["Calibration/CALIBRATION_SCALE","Calibration/SCALE",
                         "CALIBRATION/CALIBRATION_SCALE","CALIBRATION/SCALE"], "SCALE")
    offs  = read_any(h, ["Calibration/CALIBRATION_OFFSET","Calibration/OFFSET",
                         "CALIBRATION/CALIBRATION_OFFSET","CALIBRATION/OFFSET"], "OFFSET")
    return float(scale[ch_idx]), float(offs[ch_idx])

def dn_to_lin(dn, scale, offset):
    a = dn.astype(np.float64)
    a = np.where(a == 65535, np.nan, a)
    return a * scale + offset

def radiance_to_bt(L, w_um):
    L = np.maximum(L, 1e-12)
    c1, c2 = 1.19104e8, 14387.8
    with np.errstate(divide='ignore', invalid='ignore'):
        bt = (c2 / w_um) / np.log(c1/(w_um**5 * L) + 1.0)
    return np.where(np.isfinite(bt), bt, np.nan)

def robust_z(img, size=BK):
    if not np.any(np.isfinite(img)):
        return np.full_like(img, np.nan, dtype=np.float64)
    base = np.nanmedian(img)
    filled = np.where(np.isfinite(img), img, base)
    med = median_filter(filled, size=size, mode='nearest')
    mad = median_filter(np.abs(filled - med), size=size, mode='nearest')
    mad = np.where(mad < EPS_MAD, EPS_MAD, mad)
    return (filled - med) / (1.4826 * mad)

def parse_time_window(fname):
    # e.g. ..._NOM_20251004030000_20251004045959_...
    ts = re.findall(r"_(\d{14})_", os.path.basename(fname))
    if len(ts) >= 2:
        # 解析为UTC时间，然后转换为北京时间（UTC+8）
        t0_utc = datetime.strptime(ts[-2], "%Y%m%d%H%M%S")
        t1_utc = datetime.strptime(ts[-1], "%Y%m%d%H%M%S")
        # 转换为北京时间（+8小时）
        t0 = t0_utc + timedelta(hours=8)
        t1 = t1_utc + timedelta(hours=8)
        return t0, t1
    # 找单个14位时间
    ts = re.findall(r"(\d{14})", os.path.basename(fname))
    if ts:
        t_utc = datetime.strptime(ts[0], "%Y%m%d%H%M%S")
        # 转换为北京时间（+8小时）
        t = t_utc + timedelta(hours=8)
        return t, t
    return None, None

def is_fdi(f): return "_FDI-" in os.path.basename(f)
def is_geo(f): return "_GEO-" in os.path.basename(f)

# ---------------- 单景检测（返回 cand mask + T4 + geo） ----------------
def detect_once(geo_file, fdi_file):
    """
    单景火点检测
    
    参数:
        geo_file: 地理定位文件路径
        fdi_file: FDI数据文件路径
    
    返回:
        (cand, T4, lat, lon): 候选火点掩码、3.75μm亮温、纬度、经度
    """
    # 从文件名解析分辨率
    def get_resolution_from_filename(filename):
        """从文件名中提取分辨率信息"""
        # 查找常见分辨率模式，如 4000M, 2000M, 1000M, 0500M
        resolution_patterns = ["4000M", "2000M", "1000M", "0500M"]
        basename = os.path.basename(filename)
        for pattern in resolution_patterns:
            if pattern in basename:
                return pattern
        # 如果文件名中没有分辨率信息，根据数据形状推断
        return "4000M"  # 默认分辨率
    
    with h5py.File(geo_file, "r") as g:
        # 尝试读取直接的经纬度数据
        lat = read_any(g, ["Data/Latitude", "/Data/Latitude", "DATA/Latitude", "/DATA/Latitude",
                          "Latitude", "/Latitude"], "Latitude", allow_none=True)
        lon = read_any(g, ["Data/Longitude", "/Data/Longitude", "DATA/Longitude", "/DATA/Longitude",
                          "Longitude", "/Longitude"], "Longitude", allow_none=True)
        
        # 如果没有直接的经纬度数据，尝试从行列号计算
        if lat is None or lon is None:
            try:
                line_num = read_any(g, ["Navigation/LineNumber", "/Navigation/LineNumber", "NAVIGATION/LineNumber"], "LineNumber")
                col_num = read_any(g, ["Navigation/ColumnNumber", "/Navigation/ColumnNumber", "NAVIGATION/ColumnNumber"], "ColumnNumber")
                
                # 获取分辨率
                resolution = get_resolution_from_filename(geo_file)
                # 如果无法从文件名获取，根据数据形状推断
                if resolution == "4000M" and hasattr(line_num, "shape"):
                    if line_num.shape[0] > 20000:  # 0500M 分辨率约为 21984x21984
                        resolution = "0500M"
                    elif line_num.shape[0] > 10000:  # 1000M 分辨率约为 10992x10992
                        resolution = "1000M"
                    elif line_num.shape[0] > 5000:  # 2000M 分辨率约为 5496x5496
                        resolution = "2000M"
                    # 否则保持默认的4000M（约2748x2748）
                
                lat, lon = calculate_latlon(line_num, col_num, resolution)
                print(f"从行列号计算得到经纬度，形状: {lat.shape}，分辨率: {resolution}")
            except KeyError:
                print("警告: 在GEO文件中既没有找到直接的经纬度数据，也没有找到行列号数据")
                # 创建默认的经纬度网格（这只是临时解决方案，实际应用中应该报错）
                shape = (2748, 2748)  # FY-4B AGRI标准分辨率
                lat_grid = np.linspace(90, -90, shape[0])
                lon_grid = np.linspace(-180, 180, shape[1])
                lat, lon = np.meshgrid(lat_grid, lon_grid, indexing='ij')
        
        # 读取太阳天顶角
        sza = read_any(g, ["Navigation/NOMSunZenith","/Navigation/NOMSunZenith","NAVIGATION/NOMSunZenith",
                          "Data/NOMSunZenith", "/Data/NOMSunZenith", "DATA/NOMSunZenith", "/DATA/NOMSunZenith"], "SunZenith")
    if np.nanmax(sza) > 360.0:  # 有些版本是×100存的
        sza = sza / 100.0

    # 中国域掩膜
    china = (lat >= LAT_MIN) & (lat <= LAT_MAX) & (lon >= LON_MIN) & (lon <= LON_MAX)

    with h5py.File(fdi_file, "r") as f:
        dn02 = read_any(f, ["Data/NOMChannel02","DATA/NOMChannel02"], "Ch02")
        dn07 = read_any(f, ["Data/NOMChannel07","DATA/NOMChannel07"], "Ch07")
        dn13 = read_any(f, ["Data/NOMChannel13","DATA/NOMChannel13"], "Ch13")
        s02,o02 = get_calib_scale_offset(f, 1)
        s07,o07 = get_calib_scale_offset(f, 6)
        s13,o13 = get_calib_scale_offset(f, 12)

    vis = dn_to_lin(dn02, s02, o02)
    vis = np.clip(vis, 0.0, 1.2)

    L7  = dn_to_lin(dn07, s07, o07)
    L13 = dn_to_lin(dn13, s13, o13)
    T4  = radiance_to_bt(L7,  3.75)
    T13 = radiance_to_bt(L13, 10.8)
    BTD = T4 - T13

    for arr in (vis, T4, T13, BTD, sza):
        arr[~china] = np.nan

    z_btd = robust_z(BTD, size=BK)
    z_t4  = robust_z(T4,  size=BK)

    day   = (sza < DAY_SZA_MAX)
    night = (sza > NIGHT_SZA_MIN)
    cloud_day = (vis > VIS_REFL_CLOUD) & (T13 < COLD_BT13_CLOUD)

    cand_day = (day &
                (T4 >= ABS_T4_DAY) &
                (BTD >= BTD_DAY_MIN) &
                (z_btd >= Z_BTD_MIN) &
                (z_t4  >= Z_T4_MIN) &
                (~cloud_day))

    cand_ngt = (night &
                (T4 >= ABS_T4_NGT) &
                (BTD >= BTD_NGT_MIN) &
                (z_btd >= Z_BTD_MIN))

    cand = (cand_day | cand_ngt)
    cand &= np.isfinite(T4) & np.isfinite(T13) & np.isfinite(BTD)

    # 去掉孤立1像元
    lab, nlab = label(cand.astype(np.uint8))
    if nlab:
        sizes = np.bincount(lab.ravel())
        cand[np.isin(lab, np.where(sizes < 2)[0])] = False

    return cand, T4, lat, lon

# ---------------- 按天分小时聚合 ----------------
def find_pairs(data_dir, date_str, max_geo_gap_min=5):
    # 扫描顶层；若你数据在子文件夹，改成 os.walk
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(".hdf")]
    fdis  = [f for f in files if is_fdi(f) and date_str in os.path.basename(f)]
    geos  = [f for f in files if is_geo(f) and date_str in os.path.basename(f)]
    if not fdis or not geos:
        return {}

    # GEO 按结束时间排序
    geo_list = []
    for g in geos:
        g0,g1 = parse_time_window(g)
        if g0 and g1:
            geo_list.append((g,g0,g1))
    geo_list.sort(key=lambda x: x[2])

    # 为每个 FDI 找“结束<=FDI开始且最近”的 GEO
    by_hour = {}
    for f in fdis:
        f0,f1 = parse_time_window(f)
        if not (f0 and f1): continue
        best, best_gap = None, timedelta(hours=999)
        for g,g0,g1 in geo_list:
            if g1 <= f0:
                gap = f0 - g1
                if gap < best_gap:
                    best, best_gap = g, gap
        if best and best_gap <= timedelta(minutes=max_geo_gap_min):
            hour_key = f0.replace(minute=0, second=0, microsecond=0)
            by_hour.setdefault(hour_key, []).append((best, f, f0, f1))
    return by_hour

def plot_hour_png(lat, lon, freq, hour_dt, png_path):
    plt.figure(figsize=(11,9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.7)
    provinces = cfeature.NaturalEarthFeature('cultural','admin_1_states_provinces_lines','50m',facecolor='none')
    ax.add_feature(provinces, edgecolor='gray', linewidth=0.4)

    yy, xx = np.where(freq >= HITS_PERSIST_MIN)
    if len(yy):
        sc = ax.scatter(lon[yy,xx], lat[yy,xx],
                        s=np.clip((freq[yy,xx]*30), 10, 300),
                        c=freq[yy,xx], cmap="hot_r", vmin=HITS_PERSIST_MIN, vmax=np.max(freq),
                        edgecolors="k", linewidths=0.3, alpha=0.85,
                        transform=ccrs.PlateCarree())
        cb = plt.colorbar(sc, ax=ax, shrink=0.7)
        cb.set_label("Detections within hour")
    else:
        ax.text(0.5, 0.5, "No persistent hotspots", transform=ax.transAxes,
                ha="center", va="center", fontsize=16, color="red")

    plt.title(f"FY-4B Hotspots (≥{HITS_PERSIST_MIN} hits) | {hour_dt:%Y-%m-%d %H}:00")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

def run_daily(data_dir, date_str, out_dir):
    pairs_by_hour = find_pairs(data_dir, date_str)
    if not pairs_by_hour:
        print("该日期没配到 GEO/FDI 对。检查目录/日期。")
        return

    # 先检查是否有任何热点
    has_any_hotspot = False
    for hour_dt in sorted(pairs_by_hour.keys()):
        pairs = pairs_by_hour[hour_dt]
        print(f"[{hour_dt:%Y-%m-%d %H}:00] 场景数：{len(pairs)}")

        for (geo, fdi, f0, f1) in pairs:
            cand, T4, la, lo = detect_once(geo, fdi)
            if np.any(cand):
                has_any_hotspot = True
                break
        if has_any_hotspot:
            break
    
    # 如果没有热点，直接返回，不生成任何文件
    if not has_any_hotspot:
        print("该日期没有检测到任何异常热点，不生成输出文件。")
        return
    
    # 有热点才创建输出目录
    os.makedirs(out_dir, exist_ok=True)
    summary_rows = []
    png_list = []

    for hour_dt in sorted(pairs_by_hour.keys()):
        pairs = pairs_by_hour[hour_dt]
        print(f"[{hour_dt:%Y-%m-%d %H}:00] 场景数：{len(pairs)}")

        cand_list = []
        T4_list   = []
        lat = lon = None

        for (geo, fdi, f0, f1) in pairs:
            cand, T4, la, lo = detect_once(geo, fdi)
            cand_list.append(cand.astype(np.uint8))
            T4_list.append(T4)
            lat, lon = la, lo

        if not cand_list:
            continue

        freq = np.sum(cand_list, axis=0).astype(np.uint16)
        persistent = (freq >= HITS_PERSIST_MIN)

        # 每小时 CSV（持久像元）
        yy, xx = np.where(persistent)
        if len(yy):
            # 统计该小时内这些像元的最大T4
            T4_stack = np.stack([np.where(np.isfinite(t), t, -np.inf) for t in T4_list], axis=0)
            max_T4 = np.max(T4_stack[:, yy, :][:, :, :][..., range(len(xx))], axis=0) if False else np.max([t[yy,xx] for t in T4_list], axis=0)

            df = pd.DataFrame({
                "Hour": [hour_dt.strftime("%Y-%m-%d %H:00")]*len(yy),
                "Latitude":  lat[yy, xx],
                "Longitude": lon[yy, xx],
                "Hits":      freq[yy, xx],
                "Max_BT_3p75": np.max([t[yy,xx] for t in T4_list], axis=0)
            })
        else:
            df = pd.DataFrame(columns=["Hour","Latitude","Longitude","Hits","Max_BT_3p75"])

        csv_hour = os.path.join(out_dir, f"hotspots_{hour_dt:%Y%m%d_%H}00.csv")
        df.to_csv(csv_hour, index=False)

        # 统计
        summary_rows.append({
            "Hour": hour_dt.strftime("%Y-%m-%d %H:00"),
            "Scenes": len(pairs),
            "PersistentPixels": int(np.sum(persistent))
        })

        # 小时 PNG
        png_path = os.path.join(out_dir, f"map_{hour_dt:%Y%m%d_%H}00.png")
        plot_hour_png(lat, lon, freq, hour_dt, png_path)
        png_list.append(png_path)

    # 只有当有热点数据时才生成统计表
    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(os.path.join(out_dir, f"summary_{date_str}.csv"), index=False)

    # GIF - 只有当有PNG文件时才生成
    if png_list:
        gif_path = os.path.join(out_dir, f"daily_{date_str}.gif")
        if HAS_IMAGEIO:
            imgs = [imageio.imread(p) for p in png_list]
            imageio.mimsave(gif_path, imgs, duration=0.9)
        else:
            frames = [Image.open(p) for p in png_list]
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=900, loop=0)
        print(f"GIF -> {gif_path}")

    print("完成。")

# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="FY-4B | 指定日期逐小时火点聚合")
    ap.add_argument("--dir", required=True, help="含 HDF 的目录（同层放 GEO 和 FDI）")
    ap.add_argument("--date", required=True, help="目标日期，如 2025-10-20 或 20251020")
    ap.add_argument("--out",  required=True, help="输出目录")
    ap.add_argument("--hits", type=int, default=HITS_PERSIST_MIN, help="持久热点最少命中次数（默认2）")
    args = ap.parse_args()

    ds = args.date.replace("-", "")
    if len(ds) != 8:
        sys.exit("日期格式不对，用 2025-10-20 或 20251020")

    # 使用命令行参数更新常量
    HITS_PERSIST_MIN = max(1, int(args.hits))
    run_daily(args.dir, ds, args.out)