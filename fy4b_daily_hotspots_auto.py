# -*- coding: utf-8 -*-
"""
FY-4B | 指定“某一天”的逐小时火点聚合（自适应阈值版·修正版）
- 自动配对 GEO/FDI
- LUT 亮温 (CALChannel07/13)
- 角度单位自动识别（度/弧度/×100）
- 卫星天顶角掩膜 + 视场边缘屏蔽
- 正确的弧度投影公式（行列↔经纬），仅对热点像元做数值反解
- 自适应阈值（分位 + 稳健Z），零检出时逐步降阈
- 输出：每小时 CSV+PNG、全日 summary、GIF
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

try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except Exception:
    from PIL import Image
    HAS_IMAGEIO = False

# ---------------- 常量 ----------------
SUB_LON = 105.0  # FY-4B
EA = 6378.137    # km
EB = 6356.7523   # km
H  = 42164.0     # km

COFF = {"0500M": 10991.5, "1000M": 5495.5, "2000M": 2747.5, "4000M": 1373.5}
CFAC = {"0500M": 81865099, "1000M": 40932549, "2000M": 20466274, "4000M": 10233137}
LOFF = COFF
LFAC = CFAC
RES_DEFAULT = "4000M"

# 中国绘图范围
LAT_MIN, LAT_MAX = 18.0, 53.5
LON_MIN, LON_MAX = 73.0, 135.0

# 稳健Z
BK = 9
EPS_MAD = 0.3          # K，下限避免Z爆表

# 命中次数（频次）阈值
HITS_PERSIST_MIN = 1

# 日/夜（度）
DAY_SZA_MAX   = 95.0
NIGHT_SZA_MIN = 100.0

# 云判（尽量少拦）
VIS_CLOUD_REFL = 0.50
CLOUD_T13_MAX  = 280.0

# 自适应分位（更宽松）
QDAY, QNGT = 99.6, 99.3
QDAY_FALLBACK, QNGT_FALLBACK = 99.3, 99.0
MIN_VALID_PIX = 2000

# 卫星天顶角边缘屏蔽（度）
SATZEN_MAX_USE = 85.0  # >85° 视场边缘

# 连通域允许的最小像元数（1=更敏感）
MIN_BLOB_PIXELS = 1


# ---------------- 工具 ----------------
def read_any(h, paths, desc="", allow_none=False):
    for p in paths:
        if p in h:
            return h[p][:]
    if allow_none:
        return None
    raise KeyError(f"{desc} not found; tried: {paths}")

def norm_angle_deg(a, vmax_expected):
    """把角度归一到度；支持弧度/×100；65534/65535→NaN；并裁剪到[0,vmax]."""
    x = np.array(a, dtype=np.float64)
    x[(x == 65535) | (x == 65534)] = np.nan
    v95 = np.nanpercentile(x, 95) if np.any(np.isfinite(x)) else np.nan
    if np.isfinite(v95):
        if v95 <= 6.5:                 # 像弧度
            x = np.degrees(x)
        elif v95 > 2 * vmax_expected:  # 大概率×100
            x = x / 100.0
    x = np.where((x >= 0.0) & (x <= vmax_expected), x, np.nan)
    return x

def robust_z(img, size=BK):
    if not np.any(np.isfinite(img)):
        return np.full_like(img, np.nan, dtype=np.float64)
    base = np.nanmedian(img)
    filled = np.where(np.isfinite(img), img, base)
    med = median_filter(filled, size=size, mode='nearest')
    mad = median_filter(np.abs(filled - med), size=size, mode='nearest')
    mad = np.where(mad < EPS_MAD, EPS_MAD, mad)
    z = (filled - med) / (1.4826 * mad)
    return np.clip(z, -12.0, 12.0)

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

def lut_bt(f, dn_name, cal_name):
    dn  = f[dn_name][:].astype(np.int64)
    lut = f[cal_name][:].astype(np.float64)
    bt = np.full(dn.shape, np.nan, dtype=np.float64)
    m = (dn >= 0) & (dn < lut.size) & (dn != 65535)
    bt[m] = lut[dn[m]]
    return bt


# --------- 行列 <-> 经纬度（使用弧度公式；返回行/列为浮点像元坐标） ----------
def latlon2linecolumn(lat_deg, lon_deg, resolution=RES_DEFAULT, lambdaD=SUB_LON):
    if resolution not in COFF:
        raise ValueError("resolution invalid")
    λD = np.deg2rad(lambdaD)
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    eb2_ea2 = (EB**2)/(EA**2)

    φe = np.arctan(eb2_ea2 * np.tan(lat))
    cosφe = np.cos(φe)
    re = EB / np.sqrt(1 - (1 - eb2_ea2)*(cosφe**2))
    λe_λD = lon - λD

    r1 = H - re*cosφe*np.cos(λe_λD)
    r2 = -re*cosφe*np.sin(λe_λD)
    r3 = re*np.sin(φe)
    rn = np.sqrt(r1**2 + r2**2 + r3**2)

    # 关键：x,y 保持弧度（不要转度）
    x = np.arctan(-r2 / r1)
    y = np.arcsin(-r3 / rn)

    col = COFF[resolution] + x * (2 ** -16) * CFAC[resolution]
    lin = LOFF[resolution] + y * (2 ** -16) * LFAC[resolution]
    return lin, col  # 浮点像元坐标


class LC2LL_Inverter:
    """数值反解：像元(lin,col) -> (lat,lon) + 残差像元"""
    def __init__(self, resolution=RES_DEFAULT, lambdaD=SUB_LON):
        self.res = resolution
        self.lambdaD = lambdaD
        # 粗格（经纬覆盖）
        self.lat_grid = np.linspace(-60, 60, 121)
        self.lon_grid = np.linspace(lambdaD-70, lambdaD+70, 141)
        LAT, LON = np.meshgrid(self.lat_grid, self.lon_grid, indexing='ij')
        LIN, COL = latlon2linecolumn(LAT, LON, resolution=self.res, lambdaD=self.lambdaD)
        self.LIN = LIN; self.COL = COL

    def _nearest_guess(self, lin, col):
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
            h_lat = 0.05; h_lon = 0.05
            l1, c1 = latlon2linecolumn(lat + h_lat, lon, self.res, self.lambdaD)
            l2, c2 = latlon2linecolumn(lat, lon + h_lon, self.res, self.lambdaD)
            J = np.array([[l1 - f_lin, l2 - f_lin],
                          [c1 - f_col, c2 - f_col]], dtype=np.float64)
            try:
                dlat, dlon = np.linalg.solve(J, np.array([-dl, -dc], dtype=np.float64))
            except np.linalg.LinAlgError:
                break
            lat += float(dlat); lon += float(dlon)
            lat = np.clip(lat, -80.0, 80.0)
            lon = ((lon + 180.0) % 360.0) - 180.0

        rl, rc = latlon2linecolumn(lat, lon, self.res, self.lambdaD)
        resid = max(abs(rl - lin), abs(rc - col))
        return lat, lon, resid


# ---------------- 自适应阈值判别 ----------------
def adaptive_candidates(T4, T13, sza_deg, satzen_deg, vis=None,
                        q_day=QDAY, q_ngt=QNGT,
                        q_day_fb=QDAY_FALLBACK, q_ngt_fb=QNGT_FALLBACK):
    BTD = T4 - T13
    z_t4  = robust_z(T4,  size=BK)
    z_btd = robust_z(BTD, size=BK)

    day   = (sza_deg <= DAY_SZA_MAX)
    night = (sza_deg >= NIGHT_SZA_MIN)

    # 地球/边缘掩膜
    earth = np.isfinite(satzen_deg) & (satzen_deg <= SATZEN_MAX_USE)

    cloud_day = np.zeros_like(T13, dtype=bool)
    if vis is not None:
        cloud_day = (sza_deg <= 85.0) & (vis > VIS_CLOUD_REFL) & (T13 < CLOUD_T13_MAX)

    # 评分（白天偏 T4、夜间偏 BTD）
    S_day = 0.6*z_t4 + 0.4*z_btd
    S_ngt = 0.4*z_t4 + 0.6*z_btd

    valid_day = day & earth & np.isfinite(S_day) & (~cloud_day)
    valid_ngt = night & earth & np.isfinite(S_ngt)

    thr_day = np.nan; thr_ngt = np.nan
    if np.sum(valid_day) >= MIN_VALID_PIX:
        thr_day = np.nanpercentile(S_day[valid_day], q_day)
    elif np.sum(valid_day) > 0:
        thr_day = np.nanpercentile(S_day[valid_day], q_day_fb)

    if np.sum(valid_ngt) >= MIN_VALID_PIX:
        thr_ngt = np.nanpercentile(S_ngt[valid_ngt], q_ngt)
    elif np.sum(valid_ngt) > 0:
        thr_ngt = np.nanpercentile(S_ngt[valid_ngt], q_ngt_fb)

    cand_day = valid_day & (S_day >= thr_day) if np.isfinite(thr_day) else np.zeros_like(day, bool)
    cand_ngt = valid_ngt & (S_ngt >= thr_ngt) if np.isfinite(thr_ngt) else np.zeros_like(night, bool)
    cand = (cand_day | cand_ngt)
    cand &= np.isfinite(T4) & np.isfinite(T13) & np.isfinite(BTD)

    # 零检出时按阶降阈（最多 0.6）
    if not np.any(cand):
        for step in (0.2, 0.4, 0.6):
            cd = valid_day & (S_day >= (thr_day - step)) if np.isfinite(thr_day) else np.zeros_like(cand, bool)
            cn = valid_ngt & (S_ngt >= (thr_ngt - step)) if np.isfinite(thr_ngt) else np.zeros_like(cand, bool)
            cand = cd | cn
            if np.any(cand):
                break

    # 连通域去孤点（按配置）
    lab, nlab = label(cand.astype(np.uint8))
    if nlab and MIN_BLOB_PIXELS > 1:
        sizes = np.bincount(lab.ravel())
        cand[np.isin(lab, np.where(sizes < MIN_BLOB_PIXELS)[0])] = False

    return cand, dict(thr_day=float(thr_day) if np.isfinite(thr_day) else None,
                      thr_ngt=float(thr_ngt) if np.isfinite(thr_ngt) else None)


# ---------------- 单景检测 ----------------
def detect_once(geo_file, fdi_file, q_day=QDAY, q_ngt=QNGT, print_stats=False):
    with h5py.File(geo_file, "r") as g:
        sza  = read_any(g, ["Navigation/NOMSunZenith","NAVIGATION/NOMSunZenith",
                            "DATA/NOMSunZenith","Data/NOMSunZenith"], "NOMSunZenith")
        satz = read_any(g, ["Navigation/NOMSatelliteZenith","NAVIGATION/NOMSatelliteZenith",
                            "DATA/NOMSatelliteZenith","Data/NOMSatelliteZenith"], "NOMSatelliteZenith")
    sza_deg  = norm_angle_deg(sza,  180.0)
    satz_deg = norm_angle_deg(satz, 180.0)

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

    cand, thr = adaptive_candidates(T4, T13, sza_deg, satz_deg, vis, q_day=q_day, q_ngt=q_ngt)

    if print_stats:
        BTD = T4 - T13
        print("T4[K]  p50/p95/max =", np.nanpercentile(T4,50), np.nanpercentile(T4,95), np.nanmax(T4))
        print("BTD[K] p50/p95/max =", np.nanpercentile(BTD,50), np.nanpercentile(BTD,95), np.nanmax(BTD))
        print(f"SZA  min/50%/max  = {np.nanmin(sza_deg):.1f}, {np.nanpercentile(sza_deg,50):.1f}, {np.nanmax(sza_deg):.1f}")
        print(f"SatZen min/50%/max= {np.nanmin(satz_deg):.1f}, {np.nanpercentile(satz_deg,50):.1f}, {np.nanmax(satz_deg):.1f}")
        print(f"[Adaptive] thr_day={thr['thr_day']}, thr_ngt={thr['thr_ngt']}")

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
        if not (f0 and f1):
            continue
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


def plot_hour_png(lat_list, lon_list, hour_dt, png_path):
    plt.figure(figsize=(11, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.7)
    provinces = cfeature.NaturalEarthFeature('cultural','admin_1_states_provinces_lines','50m',facecolor='none')
    ax.add_feature(provinces, edgecolor='gray', linewidth=0.4)

    if lat_list:
        ax.scatter(lon_list, lat_list,
                   s=50, c='red', edgecolors="k", linewidths=0.3, alpha=0.9,
                   transform=ccrs.PlateCarree())

    plt.title(f"FY-4B Hotspots | {hour_dt:%Y-%m-%d %H}:00 (Beijing Time)")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------- 主流程 ----------------
def run_daily(data_dir, date_str, out_dir,
              hits_min=HITS_PERSIST_MIN, resolution=RES_DEFAULT, sub_lon=SUB_LON,
              q_day=QDAY, q_ngt=QNGT, max_resid_px=1.5):
    pairs_by_hour = find_pairs(data_dir, date_str)
    if not pairs_by_hour:
        print("这一天没配到 GEO/FDI 对。")
        return

    inverter = LC2LL_Inverter(resolution=resolution, lambdaD=sub_lon)

    os.makedirs(out_dir, exist_ok=True)
    summary_rows, png_list = [], []
    any_hot = False

    for hour_dt in sorted(pairs_by_hour.keys()):
        pairs = pairs_by_hour[hour_dt]
        print(f"[{hour_dt:%Y-%m-%d %H}:00] 场景数：{len(pairs)}")

        cands, T4s = [], []
        for i, (geo, fdi, f0, f1) in enumerate(pairs):
            cand, T4 = detect_once(geo, fdi, q_day=q_day, q_ngt=q_ngt, print_stats=(i==0))
            cands.append(cand.astype(np.uint8))
            T4s.append(T4)

        if not cands:
            continue

        freq = np.sum(cands, axis=0).astype(np.uint16)
        persistent = (freq >= hits_min)
        yy, xx = np.where(persistent)

        lat_list, lon_list, hits_list, maxT_list = [], [], [], []

        if len(yy) > 0:
            any_hot = True
            T4_stack = np.stack([np.where(np.isfinite(t), t, -np.inf) for t in T4s], axis=0)
            maxT4_at_pix = np.max(T4_stack[:, yy, xx], axis=0)

            for k in range(len(yy)):
                lin, col = float(yy[k]), float(xx[k])
                lat, lon, resid = inverter.invert_one(lin, col)
                if (resid <= max_resid_px) and (LAT_MIN <= lat <= LAT_MAX) and (LON_MIN <= lon <= LON_MAX):
                    lat_list.append(lat); lon_list.append(lon)
                    hits_list.append(int(freq[int(lin), int(col)]))
                    maxT_list.append(float(maxT4_at_pix[k]))

        # CSV
        csv_hour = os.path.join(out_dir, f"hotspots_{hour_dt:%Y%m%d_%H}00.csv")
        if lat_list:
            pd.DataFrame({
                "Hour": [hour_dt.strftime("%Y-%m-%d %H:00 (Beijing Time)")] * len(lat_list),
                "Latitude": lat_list, "Longitude": lon_list,
                "Hits": hits_list, "Max_BT_3p75": maxT_list
            }).to_csv(csv_hour, index=False)
        else:
            pd.DataFrame(columns=["Hour","Latitude","Longitude","Hits","Max_BT_3p75"]).to_csv(csv_hour, index=False)

        summary_rows.append({"Hour": hour_dt.strftime("%Y-%m-%d %H:00 (Beijing Time)"),
                             "Scenes": len(pairs),
                             "PersistentPixels": int(np.sum(persistent))})

        # PNG
        png_path = os.path.join(out_dir, f"map_{hour_dt:%Y%m%d_%H}00.png")
        plot_hour_png(lat_list, lon_list, hour_dt, png_path)
        png_list.append(png_path)

    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, f"summary_{date_str}.csv"), index=False)

    if png_list:
        gif_path = os.path.join(out_dir, f"daily_{date_str}.gif")
        try:
            if HAS_IMAGEIO:
                imgs = [imageio.imread(p) for p in png_list]
                imageio.mimsave(gif_path, imgs, duration=0.9)
            else:
                frames = [Image.open(p) for p in png_list]
                frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=900, loop=0)
            print(f"GIF → {gif_path}")
        except Exception as e:
            print(f"创建GIF时出错: {e}")

    if not any_hot:
        print("这一天没有检测到有效热点（自适应阈值下）。")


# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="FY-4B | 指定日期逐小时火点聚合（自适应阈值版·修正版）")
    ap.add_argument("--dir",  required=True, help="HDF 目录（同层放 GEO 与 FDI）")
    ap.add_argument("--date", required=True, help="目标日期：2025-10-20 或 20251020")
    ap.add_argument("--out",  required=True, help="输出目录")
    ap.add_argument("--hits", type=int, default=HITS_PERSIST_MIN, help="持久热点最少命中次数，默认1")
    ap.add_argument("--res",  default=RES_DEFAULT, choices=list(COFF.keys()), help="分辨率键，默认4000M")
    ap.add_argument("--sublon", type=float, default=SUB_LON, help="星下点经度，FY-4B=105.0")
    ap.add_argument("--qday", type=float, default=QDAY, help="白天 S 分位阈值，默认99.6")
    ap.add_argument("--qngt", type=float, default=QNGT, help="夜间 S 分位阈值，默认99.3")
    args = ap.parse_args()

    ds = args.date.replace("-", "")
    if len(ds) != 8:
        sys.exit("日期格式不对，用 2025-10-20 或 20251020")
    HITS_PERSIST_MIN = max(1, int(args.hits))

    run_daily(args.dir, ds, args.out,
              hits_min=HITS_PERSIST_MIN,
              resolution=args.res, sub_lon=args.sublon,
              q_day=args.qday, q_ngt=args.qngt)
