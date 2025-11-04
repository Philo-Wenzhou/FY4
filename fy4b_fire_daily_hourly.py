# -*- coding: utf-8 -*-
"""
FY-4B | 指定“某一天”的逐小时火点聚合 (稳健版)
- 自动配对 GEO/FDI
- LUT 亮温 (CALChannel07/13)
- 日/夜双阈 + 自适应兜底（分位数）
- 仅对热点像元做经纬度反求（数值法，快）
- 输出：每小时 CSV+PNG、全日 summary、GIF

依赖: h5py, numpy, pandas, scipy.ndimage, matplotlib, cartopy, imageio(或Pillow)
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

# ============ 可选 GIF ============
try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except Exception:
    from PIL import Image
    HAS_IMAGEIO = False

# ---------------- 常量 ----------------
# 几何/星下点
SUB_LON = 105.0   # FY-4B
EA = 6378.137     # km
EB = 6356.7523    # km
H  = 42164.0      # km

# 行列→角度标定常量（FY-4通用；分辨率键）
COFF = {"0500M": 10991.5, "1000M": 5495.5, "2000M": 2747.5, "4000M": 1373.5}
CFAC = {"0500M": 81865099, "1000M": 40932549, "2000M": 20466274, "4000M": 10233137}
LOFF = COFF
LFAC = CFAC
RES_DEFAULT = "4000M"

# 中国范围
LAT_MIN, LAT_MAX = 18.0, 53.5
LON_MIN, LON_MAX = 73.0, 135.0

# 滤波/稳健Z
BK = 9
EPS_MAD = 1e-10

# 聚合 ≥N 次命中才算持久热点
HITS_PERSIST_MIN = 1

# 日/夜判断（放宽暮光）
DAY_SZA_MAX   = 95.0
NIGHT_SZA_MIN = 100.0

# 阈值（经验+文献；再配自适应兜底）
ABS_T4_DAY = 335.0   # K, 白天
ABS_T4_NGT = 310.0   # K, 夜间
BTD_DAY_MIN = 6.0    # K, T4-T13
BTD_NGT_MIN = 10.0   # K
Z_BTD_MIN = 2.5
Z_T4_MIN  = 2.0

# 白天云判（可见光可用才启用）
VIS_CLOUD_REFL = 0.35
CLOUD_T13_MAX  = 285.0   # 明显白天时，T13较冷 + 反射率高 → 云

# ---------------- 小工具 ----------------
def read_any(h, paths, desc="", allow_none=False):
    for p in paths:
        if p in h:
            return h[p][:]
    if allow_none:
        return None
    raise KeyError(f"{desc} not found in file; tried: {paths}")

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
        t0_utc = datetime.strptime(ts[-2], "%Y%m%d%H%M%S")
        t1_utc = datetime.strptime(ts[-1], "%Y%m%d%H%M%S")
        return t0_utc + timedelta(hours=8), t1_utc + timedelta(hours=8)
    ts = re.findall(r"(\d{14})", os.path.basename(fname))
    if ts:
        t = datetime.strptime(ts[0], "%Y%m%d%H%M%S")
        return t + timedelta(hours=8), t + timedelta(hours=8)
    return None, None

def is_fdi(f): return "_FDI-" in os.path.basename(f)
def is_geo(f): return "_GEO-" in os.path.basename(f)

# ---------------- LUT 亮温 ----------------
def lut_bt(f, dn_name, cal_name):
    dn  = f[dn_name][:].astype(np.int64)
    lut = f[cal_name][:].astype(np.float64)
    bt = np.full(dn.shape, np.nan, dtype=np.float64)
    m = (dn >= 0) & (dn < lut.size) & (dn != 65535)
    bt[m] = lut[dn[m]]
    return bt

# ---------------- 行列 <-> 经纬度 (正解 + 反解) ----------------
def latlon2linecolumn(lat_deg, lon_deg, resolution=RES_DEFAULT, lambdaD=SUB_LON):
    """FY-4 正向投影：经纬 -> 行列 (矢量化)"""
    if resolution not in COFF:
        raise ValueError("resolution invalid")
    λD = np.deg2rad(lambdaD)
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    eb2_ea2 = (EB**2) / (EA**2)  # (b/a)^2

    φe = np.arctan(eb2_ea2 * np.tan(lat))
    cosφe = np.cos(φe)
    re = EB / np.sqrt(1 - (1 - eb2_ea2) * (cosφe**2))
    λe_λD = lon - λD

    r1 = H - re * cosφe * np.cos(λe_λD)
    r2 = -re * cosφe * np.sin(λe_λD)
    r3 = re * np.sin(φe)
    rn = np.sqrt(r1**2 + r2**2 + r3**2)

    x = np.rad2deg(np.arctan(-r2 / r1))
    y = np.rad2deg(np.arcsin(-r3 / rn))

    col = COFF[resolution] + x * (2 ** -16) * CFAC[resolution]
    lin = LOFF[resolution] + y * (2 ** -16) * LFAC[resolution]
    return np.rint(lin).astype(np.int32), np.rint(col).astype(np.int32)

# 反解：只对“热点像元 (lin,col)”做数值反求；起点来自粗格点近邻，再牛顿迭代
class LC2LL_Inverter:
    def __init__(self, resolution=RES_DEFAULT, lambdaD=SUB_LON):
        self.res = resolution
        self.lambdaD = lambdaD
        # 粗格点预制（经纬范围覆盖中国及周边）
        self.lat_grid = np.linspace(-60, 60, 121)
        self.lon_grid = np.linspace(lambdaD-70, lambdaD+70, 141)
        LAT, LON = np.meshgrid(self.lat_grid, self.lon_grid, indexing='ij')
        LIN, COL = latlon2linecolumn(LAT, LON, resolution=self.res, lambdaD=self.lambdaD)
        self.LIN = LIN; self.COL = COL

    def _nearest_guess(self, lin, col):
        # 用粗格近邻做初猜
        d2 = (self.LIN - lin)**2 + (self.COL - col)**2
        idx = np.unravel_index(np.nanargmin(d2), d2.shape)
        return float(self.lat_grid[idx[0]]), float(self.lon_grid[idx[1]])

    def invert_one(self, lin, col, max_iter=8):
        lat, lon = self._nearest_guess(lin, col)
        for _ in range(max_iter):
            f_lin, f_col = latlon2linecolumn(lat, lon, self.res, self.lambdaD)
            dl = float(f_lin - lin); dc = float(f_col - col)
            if abs(dl) < 1e-3 and abs(dc) < 1e-3:
                break
            # 数值雅可比（度→像素）；步长
            h_lat = 0.05; h_lon = 0.05
            l1, c1 = latlon2linecolumn(lat + h_lat, lon, self.res, self.lambdaD)
            l2, c2 = latlon2linecolumn(lat, lon + h_lon, self.res, self.lambdaD)
            # 雅可比矩阵
            J = np.array([[l1 - f_lin, l2 - f_lin],
                          [c1 - f_col, c2 - f_col]], dtype=np.float64)
            # 解 J * d = -[dl, dc]
            try:
                dlat, dlon = np.linalg.solve(J, np.array([-dl, -dc], dtype=np.float64))
            except np.linalg.LinAlgError:
                break
            lat += float(dlat)
            lon += float(dlon)
            # 合理范围裁剪
            lat = np.clip(lat, -80.0, 80.0)
            lon = ((lon + 180.0) % 360.0) - 180.0
        return lat, lon

# ---------------- 单景检测 ----------------
def detect_once(geo_file, fdi_file, print_stats=False):
    # GEO：SZA（经纬度缺省——稍后热点像元再反解）
    with h5py.File(geo_file, "r") as g:
        sza = read_any(g,
                       ["Navigation/NOMSunZenith","NAVIGATION/NOMSunZenith",
                        "Data/NOMSunZenith","DATA/NOMSunZenith"],
                       "SunZenith")
    sza = sza.astype(np.float64)
    if np.nanmax(sza) > 360:  # 常见 ×100
        sza = sza / 100.0

    # FDI：亮温（LUT）
    with h5py.File(fdi_file, "r") as f:
        T4  = lut_bt(f,
                     "Data/NOMChannel07" if "Data/NOMChannel07" in f else "DATA/NOMChannel07",
                     "Calibration/CALChannel07" if "Calibration/CALChannel07" in f else "CALIBRATION/CALChannel07")
        T13 = lut_bt(f,
                     "Data/NOMChannel13" if "Data/NOMChannel13" in f else "DATA/NOMChannel13",
                     "Calibration/CALChannel13" if "Calibration/CALChannel13" in f else "CALIBRATION/CALChannel13")

        vis = None
        if "Calibration/CALIBRATION_COEF(SCALE+OFFSET)" in f and \
           ("Data/NOMChannel02" in f or "DATA/NOMChannel02" in f):
            coef = f["Calibration/CALIBRATION_COEF(SCALE+OFFSET)"][:]
            dn02 = f["Data/NOMChannel02"][:] if "Data/NOMChannel02" in f else f["DATA/NOMChannel02"][:]
            s02, o02 = float(coef[1,0]), float(coef[1,1])
            vis = np.clip(dn02.astype(np.float64) * s02 + o02, 0.0, 1.2)

    BTD = T4 - T13

    if print_stats:
        print("T4[K] min/50%/95%/max =", np.nanmin(T4), np.nanpercentile(T4,50),
              np.nanpercentile(T4,95), np.nanmax(T4))
        print("BTD[K] 50%/95%/max    =", np.nanpercentile(BTD,50),
              np.nanpercentile(BTD,95), np.nanmax(BTD))
        print("SZA deg  min/50%/max  =", np.nanmin(sza), np.nanpercentile(sza,50), np.nanmax(sza))

    z_btd = robust_z(BTD, size=BK)
    z_t4  = robust_z(T4,  size=BK)

    day   = (sza <= DAY_SZA_MAX)
    night = (sza >= NIGHT_SZA_MIN)

    cloud_day = np.zeros_like(T13, dtype=bool)
    if vis is not None:
        cloud_day = (sza <= 85.0) & (vis > VIS_CLOUD_REFL) & (T13 < CLOUD_T13_MAX)

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

    # 自适应兜底：全零则用分位数启发式
    if not np.any(cand):
        qT = np.nanpercentile(T4,  99.8)
        qD = np.nanpercentile(BTD, 99.8)
        cand = (T4 >= qT) & (BTD >= qD) & np.isfinite(T4) & np.isfinite(BTD)

    # 去孤点（≥2连通像元）
    lab, nlab = label(cand.astype(np.uint8))
    if nlab:
        sizes = np.bincount(lab.ravel())
        cand[np.isin(lab, np.where(sizes < 2)[0])] = False

    return cand, T4

# ---------------- 配对/聚合 ----------------
def find_pairs(data_dir, date_str, max_geo_gap_min=5):
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
             if f.lower().endswith(".hdf")]
    fdis = [f for f in files if is_fdi(f) and date_str in os.path.basename(f)]
    geos = [f for f in files if is_geo(f) and date_str in os.path.basename(f)]
    if not fdis or not geos:
        return {}

    geo_list = []
    for g in geos:
        g0, g1 = parse_time_window(g)
        if g0 and g1:
            geo_list.append((g, g0, g1))
    geo_list.sort(key=lambda x: x[2])

    by_hour = {}
    for f in fdis:
        f0, f1 = parse_time_window(f)
        if not (f0 and f1): continue
        best, best_gap = None, timedelta(days=99)
        for g, g0, g1 in geo_list:
            if g1 <= f0:
                gap = f0 - g1
                if gap < best_gap:
                    best, best_gap = g, gap
        if best and best_gap <= timedelta(minutes=max_geo_gap_min):
            hour_key = f0.replace(minute=0, second=0, microsecond=0)
            by_hour.setdefault(hour_key, []).append((best, f, f0, f1))
    return by_hour

# ---------------- 绘图（只画热点点位） ----------------
def plot_hour_png(lat_list, lon_list, hits_list, hour_dt, png_path):
    plt.figure(figsize=(11, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.7)
    provinces = cfeature.NaturalEarthFeature('cultural','admin_1_states_provinces_lines','50m',facecolor='none')
    ax.add_feature(provinces, edgecolor='gray', linewidth=0.4)

    if len(lat_list):
        sc = ax.scatter(lon_list, lat_list,
                        s=np.clip(np.array(hits_list) * 30, 10, 300),
                        c=hits_list, cmap="hot_r", vmin=max(1, min(hits_list)),
                        vmax=max(hits_list),
                        edgecolors="k", linewidths=0.3, alpha=0.9,
                        transform=ccrs.PlateCarree())
        cb = plt.colorbar(sc, ax=ax, shrink=0.75)
        cb.set_label("Detections within hour")
    else:
        ax.text(0.5, 0.5, "No persistent hotspots", transform=ax.transAxes,
                ha="center", va="center", fontsize=16, color="red")

    plt.title(f"FY-4B Hotspots (≥{HITS_PERSIST_MIN} hits) | {hour_dt:%Y-%m-%d %H}:00")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

# ---------------- 主流程 ----------------
def run_daily(data_dir, date_str, out_dir,
              hits_min=HITS_PERSIST_MIN, resolution=RES_DEFAULT, sub_lon=SUB_LON):
    pairs_by_hour = find_pairs(data_dir, date_str)
    if not pairs_by_hour:
        print("这一天没配到 GEO/FDI 对。检查目录/日期。")
        return

    inverter = LC2LL_Inverter(resolution=resolution, lambdaD=sub_lon)

    os.makedirs(out_dir, exist_ok=True)
    summary_rows, png_list = [], []

    any_hot = False
    for hour_dt in sorted(pairs_by_hour.keys()):
        pairs = pairs_by_hour[hour_dt]
        print(f"[{hour_dt:%Y-%m-%d %H}:00] 场景数：{len(pairs)}")

        cands, T4s = [], []
        shape = None
        for i, (geo, fdi, f0, f1) in enumerate(pairs):
            cand, T4 = detect_once(geo, fdi, print_stats=(i==0))
            cands.append(cand.astype(np.uint8))
            T4s.append(T4)
            shape = cand.shape

        if not cands:
            continue

        freq = np.sum(cands, axis=0).astype(np.uint16)
        persistent = (freq >= hits_min)
        yy, xx = np.where(persistent)

        if len(yy) == 0:
            # 也输出空表，便于核对
            csv_hour = os.path.join(out_dir, f"hotspots_{hour_dt:%Y%m%d_%H}00.csv")
            pd.DataFrame(columns=["Hour","Latitude","Longitude","Hits","Max_BT_3p75"]).to_csv(csv_hour, index=False)
            summary_rows.append({"Hour": hour_dt.strftime("%Y-%m-%d %H:00"),
                                 "Scenes": len(pairs),
                                 "PersistentPixels": 0})
            png_path = os.path.join(out_dir, f"map_{hour_dt:%Y%m%d_%H}00.png")
            plot_hour_png([], [], [], hour_dt, png_path)
            png_list.append(png_path)
            continue

        any_hot = True

        # 计算这些像元的经纬度（数值反解）
        lat_list, lon_list, hits_list, maxT_list = [], [], [], []
        # 预先把各景该像元的 T4 取最大
        T4_stack = np.stack([np.where(np.isfinite(t), t, -np.inf) for t in T4s], axis=0)
        maxT4_at_pix = np.max(T4_stack[:, yy, xx], axis=0)

        for k in range(len(yy)):
            lin, col = int(yy[k]), int(xx[k])
            lat, lon = inverter.invert_one(lin, col)
            # 限制在中国范围再输出
            if (LAT_MIN <= lat <= LAT_MAX) and (LON_MIN <= lon <= LON_MAX):
                lat_list.append(lat)
                lon_list.append(lon)
                hits_list.append(int(freq[lin, col]))
                maxT_list.append(float(maxT4_at_pix[k]))

        # 小时 CSV
        csv_hour = os.path.join(out_dir, f"hotspots_{hour_dt:%Y%m%d_%H}00.csv")
        if lat_list:
            df = pd.DataFrame({
                "Hour":       [hour_dt.strftime("%Y-%m-%d %H:00")]*len(lat_list),
                "Latitude":   lat_list,
                "Longitude":  lon_list,
                "Hits":       hits_list,
                "Max_BT_3p75": maxT_list
            })
        else:
            df = pd.DataFrame(columns=["Hour","Latitude","Longitude","Hits","Max_BT_3p75"])
        df.to_csv(csv_hour, index=False)

        summary_rows.append({"Hour": hour_dt.strftime("%Y-%m-%d %H:00"),
                             "Scenes": len(pairs),
                             "PersistentPixels": int(np.sum(persistent))})

        # 小时 PNG
        png_path = os.path.join(out_dir, f"map_{hour_dt:%Y%m%d_%H}00.png")
        plot_hour_png(lat_list, lon_list, hits_list, hour_dt, png_path)
        png_list.append(png_path)

    # 全日统计 & GIF
    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, f"summary_{date_str}.csv"), index=False)

    if png_list and len(png_list) > 0:
        gif_path = os.path.join(out_dir, f"daily_{date_str}.gif")
        try:
            if HAS_IMAGEIO:
                # 确保所有图像具有相同的尺寸
                imgs = []
                target_size = None
                for p in png_list:
                    img = imageio.imread(p)
                    if target_size is None:
                        target_size = img.shape
                    # 如果图像尺寸不一致，调整为第一个图像的尺寸
                    if img.shape != target_size:
                        # 使用PIL调整尺寸
                        from PIL import Image
                        pil_img = Image.fromarray(img)
                        # 计算调整后的尺寸（保持比例）
                        width, height = pil_img.size
                        target_width, target_height = target_size[1], target_size[0]
                        # 保持比例缩放
                        scale = min(target_width/width, target_height/height)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        resized = pil_img.resize((new_width, new_height), Image.LANCZOS)
                        # 创建目标尺寸的空白图像并居中放置
                        final_img = Image.new('RGB', (target_width, target_height), (255, 255, 255))
                        offset = ((target_width - new_width) // 2, (target_height - new_height) // 2)
                        final_img.paste(resized, offset)
                        img = np.array(final_img)
                    imgs.append(img)
                # 保存GIF
                imageio.mimsave(gif_path, imgs, duration=0.9)
            else:
                # 使用PIL时也需要确保尺寸一致
                frames = []
                target_size = None
                for p in png_list:
                    frame = Image.open(p)
                    if target_size is None:
                        target_size = frame.size
                    # 如果图像尺寸不一致，调整为第一个图像的尺寸
                    if frame.size != target_size:
                        # 保持比例缩放
                        scale = min(target_size[0]/frame.size[0], target_size[1]/frame.size[1])
                        new_width = int(frame.size[0] * scale)
                        new_height = int(frame.size[1] * scale)
                        resized = frame.resize((new_width, new_height), Image.LANCZOS)
                        # 创建目标尺寸的空白图像并居中放置
                        final_img = Image.new('RGB', target_size, (255, 255, 255))
                        offset = ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2)
                        final_img.paste(resized, offset)
                        frame = final_img
                    frames.append(frame)
                # 保存GIF
                if frames:
                    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=900, loop=0)
            print(f"GIF → {gif_path}")
        except Exception as e:
            print(f"创建GIF时出错: {e}")
            # 不中断程序，继续执行后续代码

    if not any_hot:
        print("这一天没有检测到有效热点（按当前阈值/规则）。已输出空表+底图。")

# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="FY-4B | 指定日期逐小时火点聚合（稳健版）")
    ap.add_argument("--dir",  required=True, help="HDF 目录（同层放 GEO 与 FDI）")
    ap.add_argument("--date", required=True, help="目标日期：2025-10-20 或 20251020")
    ap.add_argument("--out",  required=True, help="输出目录")
    ap.add_argument("--hits", type=int, default=HITS_PERSIST_MIN, help="持久热点最少命中次数，默认1")
    ap.add_argument("--res",  default=RES_DEFAULT, choices=list(COFF.keys()), help="分辨率键，默认4000M")
    ap.add_argument("--sublon", type=float, default=SUB_LON, help="星下点经度，FY-4B=105.0")
    args = ap.parse_args()

    ds = args.date.replace("-", "")
    if len(ds) != 8:
        sys.exit("日期格式不对，用 2025-10-20 或 20251020")

    HITS_PERSIST_MIN = max(1, int(args.hits))
    run_daily(args.dir, ds, args.out, hits_min=HITS_PERSIST_MIN,
              resolution=args.res, sub_lon=args.sublon)
