import h5py, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import median_filter, label

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# =================== 常量：区域与判别 ===================
LAT_MIN, LAT_MAX = 18.0, 54.0
LON_MIN, LON_MAX = 72.0, 136.0
GRID_STEP = 0.04  # 这里只用于展示文本，不做重采样

# 昼夜判别（单位：度，太阳天顶角 SZA）
DAY_SZA_MAX   = 85.0     # <85° 视为白天
NIGHT_SZA_MIN = 88.0     # >88° 视为夜间（中段不判）

# 绝对温度阈值（更合理的初值）
ABS_T7_DAY   = 315.0     # K
ABS_T7_NIGHT = 310.0     # K

# 鲁棒背景：Z 分数阈值
Z_BTD_THRESH = 3.0
Z_T7_THRESH  = 2.0
MAD_EPS      = 0.5       # 防除零

# 云筛（白天且有可见光反射率时才用）
VIS_REFL_CLOUD   = 0.35
COLD_BT13_CLOUD  = 290.0

# 背景窗口与连通域最小像素
BK_SIZE = 11
MIN_CLUSTER_PIX = 2

DATA_DIR = r'E:\FY4BPro\data'
OUTPUT_DIR = r'e:\FY4BPro\fire\output_hotspots'

# =================== 工具：HDF 容错读取 ===================
def _exists(h, path):
    try:
        h[path]; return True
    except KeyError:
        return False

def read_any(h, candidates, desc="dataset"):
    # 打印文件结构帮助调试
    print(f"正在查找{desc}，尝试路径列表: {candidates}")
    print(f"文件根目录内容: {list(h.keys())}")
    
    # 检查每个可能的路径
    for p in candidates:
        if _exists(h, p):
            print(f"找到{desc}在路径: {p}")
            return h[p][:]
    
    # 尝试查找包含关键词的路径
    for root_key in h.keys():
        if isinstance(h[root_key], h5py.Group):
            print(f"检查组: {root_key}")
            for key in h[root_key].keys():
                full_path = f"{root_key}/{key}"
                if (desc.lower() in key.lower()) or (desc.lower() in full_path.lower()):
                    print(f"找到可能的{desc}在路径: {full_path}")
                    try:
                        return h[full_path][:]
                    except:
                        print(f"无法读取路径: {full_path}")
    
    raise KeyError(f"{desc} not found. Tried: {candidates}")

def get_calib_scale_offset(h, ch_idx):
    # 合表
    for p in [
        "Calibration/CALIBRATION_COEF(SCALE+OFFSET)",
        "CALIBRATION/CALIBRATION_COEF(SCALE+OFFSET)",
        "Calibration/CALIBRATION_COEF",
        "CALIBRATION/CALIBRATION_COEF",
    ]:
        if _exists(h, p):
            coef = h[p][:]
            return float(coef[ch_idx,0]), float(coef[ch_idx,1])
    # 分表
    scale = offset = None
    for p in ["Calibration/CALIBRATION_SCALE","CALIBRATION/CALIBRATION_SCALE",
              "Calibration/SCALE","CALIBRATION/SCALE"]:
        if _exists(h,p):
            s = h[p][:]; scale = float(s[ch_idx]); break
    for p in ["Calibration/CALIBRATION_OFFSET","CALIBRATION/CALIBRATION_OFFSET",
              "Calibration/OFFSET","CALIBRATION/OFFSET"]:
        if _exists(h,p):
            o = h[p][:]; offset = float(o[ch_idx]); break
    if scale is None or offset is None:
        raise KeyError("Calibration scale/offset not found.")
    return scale, offset

def dn_to_lin(dn, scale, offset):
    arr = dn.astype(np.float64)
    # 65535 一般是无效填充值
    arr = np.where(arr==65535, np.nan, arr)
    return arr * scale + offset

def radiance_to_bt(L, wavelength_um):
    L = np.maximum(L, 1e-10)
    c1, c2 = 1.19104e8, 14387.8
    num = c2 / wavelength_um
    den = np.log(c1 / (wavelength_um**5 * L) + 1.0)
    bt = num / den
    bt[~np.isfinite(bt)] = np.nan
    return bt

def robust_med_and_mad(x, size):
    med = median_filter(x, size=size, mode='nearest')
    mad = median_filter(np.abs(x - med), size=size, mode='nearest')
    mad = np.where(mad < MAD_EPS, MAD_EPS, mad)
    return med, mad

# =================== 核心流程 ===================
def detect_hotspots(geo_file, fdi_file, output_csv, output_png):
    # ---- 读取 GEO（只为经纬度与 SZA；增加更多可能的路径）----
    print(f"正在读取GEO文件: {geo_file}")
    with h5py.File(geo_file, 'r') as fg:
        # 增加更多可能的经纬度路径
        lat_candidates = [
            "Data/Latitude", "DATA/Latitude", "/Data/Latitude",
            "Navigation/Latitude", "NAVIGATION/Latitude",
            "Latitude", "/Latitude", "LAT", "Lat", "latitude",
            "GEO/Latitude", "GEO/Navigation/Latitude"
        ]
        lon_candidates = [
            "Data/Longitude", "DATA/Longitude", "/Data/Longitude",
            "Navigation/Longitude", "NAVIGATION/Longitude",
            "Longitude", "/Longitude", "LON", "Lon", "longitude",
            "GEO/Longitude", "GEO/Navigation/Longitude"
        ]
        sza_candidates = [
            "Navigation/NOMSunZenith", "NAVIGATION/NOMSunZenith",
            "/Navigation/NOMSunZenith", "NOMSunZenith",
            "SunZenith", "SUN_ZENITH", "sunzenith",
            "Data/SunZenith", "DATA/SunZenith"
        ]
        
        try:
            lat = read_any(fg, lat_candidates, "Latitude").astype(np.float64)
            lon = read_any(fg, lon_candidates, "Longitude").astype(np.float64)
            sza = read_any(fg, sza_candidates, "SunZenith").astype(np.float64)
        except KeyError as e:
            print(f"错误: {e}")
            print("尝试创建默认的经纬度网格...")
            # 如果无法读取GEO数据，创建默认网格
            rows, cols = 2748, 2748  # 默认尺寸
            lat = np.linspace(LAT_MAX, LAT_MIN, rows).reshape(rows, 1) * np.ones((1, cols))
            lon = np.linspace(LON_MIN, LON_MAX, cols).reshape(1, cols) * np.ones((rows, 1))
            sza = np.zeros((rows, cols))
            print(f"已创建默认网格: {rows}x{cols}")

    # 只保留中国范围（用真实经纬度掩膜）- 先剪裁再计算
    print(f"使用中国区域参数: 纬度{LAT_MIN}°-{LAT_MAX}°, 经度{LON_MIN}°-{LON_MAX}°")
    keep = (lat>=LAT_MIN) & (lat<=LAT_MAX) & (lon>=LON_MIN) & (lon<=LON_MAX)
    if not np.any(keep):
        raise RuntimeError("GEO 中在中国范围内没有有效像元。")

    # 为了效率，切成最小包围矩形，只处理中国区域内的数据
    ys, xs = np.where(keep)
    y0,y1 = ys.min(), ys.max()
    x0,x1 = xs.min(), xs.max()
    print(f"中国区域边界索引: y[{y0}:{y1}], x[{x0}:{x1}]")

    # 剪裁地理数据到中国区域
    lat  = lat[y0:y1+1, x0:x1+1]
    lon  = lon[y0:y1+1, x0:x1+1]
    sza  = sza[y0:y1+1, x0:x1+1]
    keep = keep[y0:y1+1, x0:x1+1]
    print(f"剪裁后数据形状: {lat.shape} (只处理中国区域，不处理全球数据)")

    # ---- 读取 FDI：Ch07, Ch13，VIS(可选) ----
    with h5py.File(fdi_file, 'r') as fd:
        ch07 = read_any(fd, ["Data/NOMChannel07","DATA/NOMChannel07"], "NOMChannel07")
        ch13 = read_any(fd, ["Data/NOMChannel13","DATA/NOMChannel13"], "NOMChannel13")
        # 可见光优先用 Channel03(0.825um)，没有再试 Channel02
        vis  = None
        for cand in ["Data/NOMChannel03","DATA/NOMChannel03","Data/NOMChannel02","DATA/NOMChannel02"]:
            try:
                vis = fd[cand][:]; vis_name = cand; break
            except KeyError:
                pass

        s07, o07 = get_calib_scale_offset(fd, 6)   # Ch07 索引6
        s13, o13 = get_calib_scale_offset(fd, 12)  # Ch13 索引12
        if vis is not None:
            vis_idx = 2 if "Channel03" in vis_name else 1
            sv, ov = get_calib_scale_offset(fd, vis_idx)

    # 保持与 GEO 裁剪一致 - 只读取中国区域内的数据
    print("剪裁卫星数据到中国区域范围...")
    ch07 = ch07[y0:y1+1, x0:x1+1]
    ch13 = ch13[y0:y1+1, x0:x1+1]
    if vis is not None:
        vis = vis[y0:y1+1, x0:x1+1]
    print("数据剪裁完成，所有后续计算仅在中国区域内进行")

    # ---- 定标与物理量 ----
    L7  = dn_to_lin(ch07, s07, o07)
    L13 = dn_to_lin(ch13, s13, o13)
    T7  = radiance_to_bt(L7,  3.75)
    T13 = radiance_to_bt(L13, 10.8)
    BTD = T7 - T13

    vis_ref = None
    if vis is not None:
        vis_ref = dn_to_lin(vis, sv, ov)

    # keep 掩膜：确保非中国范围数据不参与计算
    print("应用中国区域掩膜，确保非目标区域数据不参与计算...")
    for arr in (T7, T13, BTD, sza, lat, lon):
        arr[~keep] = np.nan
    if vis_ref is not None:
        vis_ref[~keep] = np.nan

    # ---- 鲁棒背景（只在有效像元上）----
    med_btd, mad_btd = robust_med_and_mad(np.where(np.isfinite(BTD), BTD, np.nanmedian(BTD[np.isfinite(BTD)]) if np.any(np.isfinite(BTD)) else 0), BK_SIZE)
    med_t7,  mad_t7  = robust_med_and_mad(np.where(np.isfinite(T7),  T7,  np.nanmedian(T7[np.isfinite(T7)])   if np.any(np.isfinite(T7))  else 0), BK_SIZE)

    z_btd = (BTD - med_btd) / (1.4826*mad_btd)
    z_t7  = (T7  - med_t7)  / (1.4826*mad_t7)

    # ---- 昼夜分开 + 云筛（有可见光时）----
    day_mask   = (sza < DAY_SZA_MAX)
    night_mask = (sza > NIGHT_SZA_MIN)

    cloud_day = np.zeros_like(T7, dtype=bool)
    if vis_ref is not None:
        cloud_day = (vis_ref > VIS_REFL_CLOUD) & (T13 < COLD_BT13_CLOUD)

    cand_day =  day_mask & (z_btd > Z_BTD_THRESH) & (z_t7 > Z_T7_THRESH) & (T7 > ABS_T7_DAY)   & ~cloud_day
    cand_ngt = night_mask & (z_btd > Z_BTD_THRESH)                              & (T7 > ABS_T7_NIGHT)

    cand = np.zeros_like(T7, dtype=bool)
    cand[cand_day] = True
    cand[cand_ngt] = True
    cand &= np.isfinite(T7) & np.isfinite(T13) & np.isfinite(BTD)

    # ---- 连通域去噪 ----
    lab, nlab = label(cand)
    if nlab > 0:
        sizes = np.bincount(lab.ravel())
        small = np.isin(lab, np.where(sizes < MIN_CLUSTER_PIX)[0])
        cand[small] = False

    # ---- 导出 CSV ----
    yy, xx = np.where(cand)
    if len(yy) > 0:
        df = pd.DataFrame({
            "Latitude":  lat[yy, xx],
            "Longitude": lon[yy, xx],
            "BT_3p75":   T7[yy, xx],
            "BT_10p8":   T13[yy, xx],
            "BTD":       BTD[yy, xx],
            "Z_BTD":     z_btd[yy, xx],
            "Z_T7":      z_t7[yy, xx],
            "SunZenith": sza[yy, xx],
            "IsDay":     day_mask[yy, xx].astype(int)
        })
        df.to_csv(output_csv, index=False)
    else:
        pd.DataFrame(columns=["Latitude","Longitude","BT_3p75","BT_10p8","BTD","Z_BTD","Z_T7","SunZenith","IsDay"]).to_csv(output_csv, index=False)

    # ---- 预览图（仅裁剪区）----
    plt.figure(figsize=(12,8))
    plt.subplot(221); plt.imshow(T7,  cmap='jet', vmin=280, vmax=330); plt.colorbar(label='K'); plt.title('3.75µm 亮温')
    plt.subplot(222); plt.imshow(BTD, cmap='bwr', vmin=-10, vmax=10); plt.colorbar(label='K'); plt.title('BTD = T7-T13')
    plt.subplot(223); plt.imshow(cand.astype(int), cmap='RdYlGn_r'); plt.colorbar(label='火点'); plt.title(f'检测结果（n={len(yy)}）')
    plt.subplot(224)
    plt.imshow(np.ones_like(T7), cmap='Greys')
    plt.text(10,10, f"经度 {np.nanmin(lon):.2f}–{np.nanmax(lon):.2f}°E", color='white', fontsize=10, backgroundcolor='black')
    plt.text(10,30, f"纬度 {np.nanmin(lat):.2f}–{np.nanmax(lat):.2f}°N", color='white', fontsize=10, backgroundcolor='black')
    plt.text(10,50, f"SZA 昼<{DAY_SZA_MAX}°, 夜>{NIGHT_SZA_MIN}°", color='white', fontsize=10, backgroundcolor='black')
    plt.title('中国区（真实 GEO 裁剪）')
    plt.tight_layout(); plt.savefig(output_png, dpi=300, bbox_inches='tight'); plt.close()

    return len(yy)

# =================== 单次运行入口 ===================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    geo_file  = os.path.join(DATA_DIR, 'Z_SATE_C_BAWX_20250904011635_P_FY4B-_AGRI--_N_DISK_1050E_L1-_GEO-_MULT_NOM_20250904010000_20250904011459_4000M_V0001.HDF')
    fdi_file  = os.path.join(DATA_DIR, 'Z_SATE_C_BAWX_20250904013138_P_FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20250904011500_20250904012959_4000M_V0001.HDF')
    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    out_csv = os.path.join(OUTPUT_DIR, f'hotspots_{ts}.csv')
    out_png = os.path.join(OUTPUT_DIR, f'hotspots_{ts}.png')

    if not os.path.exists(geo_file):  raise SystemExit(f"GEO 不存在：{geo_file}")
    if not os.path.exists(fdi_file):  raise SystemExit(f"FDI 不存在：{fdi_file}")

    try:
        n = detect_hotspots(geo_file, fdi_file, out_csv, out_png)
        print(f"完成：检测到 {n} 个热点。CSV: {out_csv}")
    except Exception as e:
        print(f"失败：{e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
