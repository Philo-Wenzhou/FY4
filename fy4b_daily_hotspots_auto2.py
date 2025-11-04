# -*- coding: utf-8 -*-
"""
FY-4B | 指定“某一天”的逐小时火点聚合（情境阈值 + 可选Dozier反演·更敏版）
- 自动配对 GEO/FDI
- LUT 亮温 (CALChannel07/13)
- 文献法：潜在火像元(PFP) + 局地背景(21x21/31x31) + 情境检验（昼/夜分开）
- 灵敏度旋钮 (--sens)：<1 更敏；>1 更保守；默认 0.85（更敏）
- 可选 Dozier 亚像元反演 (--dozier 1)：估计火温 Tf 与面积份额 f（3.9µm & 10.8µm）
- 仅对热点像元做经纬度反求（数值法 + 回投残差过滤）
- 输出：每小时 CSV(+可选Tf,f)+PNG、全日 summary、GIF
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

# 中国范围
LAT_MIN, LAT_MAX = 18.0, 53.5
LON_MIN, LON_MAX = 73.0, 135.0

# 稳健参数
BK = 9
EPS_MAD = 0.3            # MAD 下限 (K)
SATZEN_MAX_USE = 85.0    # 视场边缘屏蔽（必要时可放宽到 87）
MIN_BLOB_PIXELS = 1      # 连通域最小像元

# 日/夜划分（度）
DAY_SZA_MAX   = 95.0
NIGHT_SZA_MIN = 100.0

# 白天云判（尽量少拦）
VIS_CLOUD_REFL = 0.50
CLOUD_T13_MAX  = 280.0

# 情境窗口与背景像元阈值
CTX_WIN_LIST   = [21, 31]    # 先 21x21，不足再 31x31
CTX_MIN_BG_PIX = 80

# MOD14/VIIRS 风格阈值（已整体下调一档，仍可随 sens 缩放）
ABS_T4_DAY   = 312.0   # K （原 315）
ABS_T4_NGT   = 307.0   # K （原 310）
ABS_BTD_DAY  = 8.0     # K （原 10）
ABS_BTD_NGT  = 6.0     # K （原 8）

K_T4_DAY   = 3.0       # 背景 σ 倍数（昼）
K_T4_NGT   = 2.5
K_BTD_DAY  = 3.0
K_BTD_NGT  = 2.5

OFFS_T4_DAY  = 3.0     # K：最小增量下限（避免σ很小时误报）
OFFS_T4_NGT  = 2.0
OFFS_BTD_DAY = 3.5
OFFS_BTD_NGT = 2.5

# Sun glint（若可用则加严以防假警；否则忽略）
GLINT_THR_DEG = 36.0

# 聚合
HITS_PERSIST_MIN = 1

# 物理常数（Dozier用）
C1 = 1.191042e-16   # W·m^-2·sr^-1·m^3
C2 = 1.4387752e-2   # m·K

# ---------------- 工具 ----------------
def read_any(h, paths, desc="", allow_none=False):
    for p in paths:
        if p in h: return h[p][:]
    if allow_none: return None
    raise KeyError(f"{desc} not found; tried: {paths}")

def norm_angle_deg(a, vmax_expected):
    """自动把角度归一到度。支持：度、弧度、百分度（×100）。65534/65535→NaN"""
    x = np.array(a, dtype=np.float64)
    x[(x == 65535) | (x == 65534)] = np.nan
    v95 = np.nanpercentile(x, 95) if np.any(np.isfinite(x)) else np.nan
    if np.isfinite(v95):
        if v95 <= 6.5:                  # radians
            x = np.degrees(x)
        elif v95 > 2 * vmax_expected:   # ×100
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

# --------- 行列 <-> 经纬度（浮点） ----------
def latlon2linecolumn(lat_deg, lon_deg, resolution=RES_DEFAULT, lambdaD=SUB_LON):
    if resolution not in COFF:
        raise ValueError("resolution invalid")
    λD = np.deg2rad(lambdaD)
    lat = np.deg2rad(lat_deg); lon = np.deg2rad(lon_deg)
    eb2_ea2 = (EB**2)/(EA**2)
    φe = np.arctan(eb2_ea2 * np.tan(lat))
    cosφe = np.cos(φe)
    re = EB / np.sqrt(1 - (1 - eb2_ea2)*(cosφe**2))
    λe_λD = lon - λD
    r1 = H - re*cosφe*np.cos(λe_λD)
    r2 = -re*cosφe*np.sin(λe_λD)
    r3 = re*np.sin(φe)
    rn = np.sqrt(r1**2 + r2**2 + r3**2)
    x = np.rad2deg(np.arctan(-r2/r1))
    y = np.rad2deg(np.arcsin(-r3/rn))
    col = COFF[resolution] + x*(2**-16)*CFAC[resolution]
    lin = LOFF[resolution] + y*(2**-16)*LFAC[resolution]
    return lin, col   # 浮点

class LC2LL_Inverter:
    def __init__(self, resolution=RES_DEFAULT, lambdaD=SUB_LON):
        self.res = resolution; self.lambdaD = lambdaD
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
            J = np.array([[l1 - f_lin, l2 - f_lin],[c1 - f_col, c2 - f_col]], dtype=np.float64)
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

# ---------------- 文献法：潜在火像元 + 情境阈值 ----------------
def _local_bg_stats(arr_list, i, j, win):
    """从多张栈(arr_list)里抽取窗口，返回 [nwin, n_layers]（含NaN）"""
    Hh = win // 2
    ys = max(0, i - Hh); ye = min(arr_list[0].shape[0], i + Hh + 1)
    xs = max(0, j - Hh); xe = min(arr_list[0].shape[1], j + Hh + 1)
    vals = []
    for A in arr_list:
        sub = A[ys:ye, xs:xe]
        vals.append(sub.reshape(-1))
    V = np.vstack(vals).T
    return V

def contextual_candidates(T4, T13, sza_deg, satz_deg, sun_glint=None, vis=None, sens=0.85, debug=False):
    """
    MOD14/VIIRS 思路：
    1) PFP 潜在火像元（绝对阈值 OR 鲁棒Z-OR；注意：z 判定使用 OR）
    2) 对每个PFP，计算局地背景（21x21不足则31x31）
    3) 昼/夜分别做 情境检验（绝对增量 OR σ倍数+下限）
    4) 云与视角屏蔽，glint时加严
    """
    assert 0.3 <= sens <= 2.0, "sens 建议在[0.3,2.0]内"

    BTD = T4 - T13
    z_t4  = robust_z(T4)
    z_btd = robust_z(BTD)

    # 基础掩膜
    earth = np.isfinite(satz_deg) & (satz_deg <= SATZEN_MAX_USE)
    day   = (sza_deg <= DAY_SZA_MAX)
    night = (sza_deg >= NIGHT_SZA_MIN)

    cloud_day = np.zeros_like(T13, dtype=bool)
    if vis is not None:
        cloud_day = (sza_deg <= 85.0) & (vis > VIS_CLOUD_REFL) & (T13 < CLOUD_T13_MAX)

    # 1) 潜在火像元（更敏感：绝对阈值与z阈值按 sens 缩放；z 用 OR）
    abs_t4_day  = ABS_T4_DAY  * (1.00 * sens)
    abs_t4_ngt  = ABS_T4_NGT  * (1.00 * sens)
    abs_dtd_day = ABS_BTD_DAY * (1.00 * sens)
    abs_dtd_ngt = ABS_BTD_NGT * (1.00 * sens)

    z_t4_thr  = 2.5 * (1.00 * sens)
    z_btd_thr = 2.0 * (1.00 * sens)
    z_hit = (z_t4 >= z_t4_thr) | (z_btd >= z_btd_thr)

    pfp = np.zeros_like(T4, dtype=bool)
    pfp |= (day   & earth & ((T4 >= abs_t4_day) | (BTD >= abs_dtd_day) | z_hit) & (~cloud_day))
    pfp |= (night & earth & ((T4 >= abs_t4_ngt) | (BTD >= abs_dtd_ngt) | z_hit))

    # 2)+3) 情境检验
    Hn, Wn = T4.shape
    cand = np.zeros_like(T4, dtype=bool)

    k_t4_day  = K_T4_DAY  / sens
    k_t4_ngt  = K_T4_NGT  / sens
    k_btd_day = K_BTD_DAY / sens
    k_btd_ngt = K_BTD_NGT / sens

    off_t4_day  = OFFS_T4_DAY  * sens**0.5
    off_t4_ngt  = OFFS_T4_NGT  * sens**0.5
    off_btd_day = OFFS_BTD_DAY * sens**0.5
    off_btd_ngt = OFFS_BTD_NGT * sens**0.5

    glint_mask = np.zeros_like(T4, dtype=bool)
    if sun_glint is not None:
        glint_mask = (sun_glint <= GLINT_THR_DEG)

    stack_for_bg = [
        np.where(np.isfinite(T4),  T4,  np.nan),
        np.where(np.isfinite(T13), T13, np.nan),
        np.where(np.isfinite(BTD), BTD, np.nan)
    ]

    yy, xx = np.where(pfp)
    for y, x in zip(yy, xx):
        V = None
        used_win = None
        for win in CTX_WIN_LIST:
            V = _local_bg_stats(stack_for_bg, y, x, win)  # [nwin, 3]
            t4v  = V[:,0]; t13v = V[:,1]; btdv = V[:,2]
            m_bg = np.isfinite(t4v) & np.isfinite(t13v) & np.isfinite(btdv)
            if np.sum(m_bg) >= CTX_MIN_BG_PIX:
                used_win = win
                break
        if used_win is None:
            continue

        t4_bg  = t4v[m_bg]; btd_bg = btdv[m_bg]
        q99_t4  = np.percentile(t4_bg,  99.0)
        q99_btd = np.percentile(btd_bg, 99.0)
        t4_bg_w  = np.clip(t4_bg,  None, q99_t4)
        btd_bg_w = np.clip(btd_bg, None, q99_btd)

        mu_t4  = float(np.mean(t4_bg_w));  sd_t4  = float(np.std(t4_bg_w, ddof=1) + 1e-6)
        mu_btd = float(np.mean(btd_bg_w)); sd_btd = float(np.std(btd_bg_w, ddof=1) + 1e-6)

        t4_xy   = float(T4[y, x]);   btd_xy  = float(BTD[y, x])
        day_xy  = bool(day[y, x]);   night_xy = bool(night[y, x])
        cloud_xy= bool(cloud_day[y, x]) if day_xy else False
        glint_xy= bool(glint_mask[y, x])

        pass_ctx = False
        if day_xy and (not cloud_xy):
            inc_t4  = t4_xy  - mu_t4
            inc_btd = btd_xy - mu_btd
            thr_t4  = max(k_t4_day  * sd_t4,  off_t4_day)
            thr_btd = max(k_btd_day * sd_btd, off_btd_day)
            if glint_xy:
                thr_t4  *= 1.2
                thr_btd *= 1.2
                pass_ctx = (inc_t4 >= thr_t4) and (inc_btd >= thr_btd)
            else:
                pass_ctx = (inc_t4 >= thr_t4) or  (inc_btd >= thr_btd)
            pass_ctx |= (t4_xy >= abs_t4_day) or (btd_xy >= abs_dtd_day)

        elif night_xy:
            inc_t4  = t4_xy  - mu_t4
            inc_btd = btd_xy - mu_btd
            thr_t4  = max(k_t4_ngt  * sd_t4,  off_t4_ngt)
            thr_btd = max(k_btd_ngt * sd_btd, off_btd_ngt)
            pass_ctx = (inc_t4 >= thr_t4) or (inc_btd >= thr_btd)
            pass_ctx |= (t4_xy >= abs_t4_ngt) or (btd_xy >= abs_dtd_ngt)

        if pass_ctx:
            cand[y, x] = True

    # 连通域过滤
    if MIN_BLOB_PIXELS > 1:
        lab, nlab = label(cand.astype(np.uint8))
        if nlab:
            sizes = np.bincount(lab.ravel())
            cand[np.isin(lab, np.where(sizes < MIN_BLOB_PIXELS)[0])] = False

    # ——兜底：若 cand 为空，回退到 PFP（防“0 报警”）
    if not np.any(cand):
        cand = pfp.copy()

    if debug:
        print("[DBG] earth=", np.sum(earth),
              " day=", np.sum(day & earth),
              " night=", np.sum(night & earth),
              " pfp=", np.sum(pfp),
              " cand=", np.sum(cand))

    return cand

# ---------------- Dozier 亚像元反演（可选） ----------------
def planck_radiance_um(T, um):
    """近似单波段普朗克辐射亮度 (W·m^-2·sr^-1·µm^-1)"""
    lam = um * 1e-6
    L = (C1 / (lam**5)) / (np.exp(C2 / (lam * (T + 1e-9))) - 1.0)  # per m
    return L * 1e-6  # per µm

def dozier_estimate(T4, T13, mu_T4_bg, mu_T13_bg, emiss_39=0.95, emiss_108=0.98):
    """
    简化 Dozier：L_meas = (1-f)*L_bg + f*ε*L(Tf)
    用两个波段（3.9µm与10.8µm）求 f 与 Tf（网格搜索 Tf，再一致化 f）。
    返回 (Tf[K], f[0-1])；失败返回 (np.nan, np.nan)
    """
    if not (np.isfinite(T4) and np.isfinite(T13) and np.isfinite(mu_T4_bg) and np.isfinite(mu_T13_bg)):
        return np.nan, np.nan

    L4_meas  = planck_radiance_um(T4, 3.9)
    L13_meas = planck_radiance_um(T13, 10.8)
    L4_bg    = planck_radiance_um(mu_T4_bg, 3.9)
    L13_bg   = planck_radiance_um(mu_T13_bg, 10.8)

    Tf_grid = np.linspace(500.0, 1500.0, 401)  # 500–1500K
    L4_fire = planck_radiance_um(Tf_grid, 3.9) * emiss_39
    L13_fire= planck_radiance_um(Tf_grid,10.8) * emiss_108

    f4  = (L4_meas  - L4_bg)  / np.maximum(L4_fire  - L4_bg, 1e-12)
    f13 = (L13_meas - L13_bg) / np.maximum(L13_fire - L13_bg, 1e-12)

    diff = np.abs(f4 - f13)
    mask = (f4 > 0.0) & (f13 > 0.0) & (f4 <= 1.0) & (f13 <= 1.0)
    if not np.any(mask):
        return np.nan, np.nan
    idx = np.argmin(np.where(mask, diff, np.inf))
    Tf = float(Tf_grid[idx]); f  = float(0.5 * (f4[idx] + f13[idx]))
    return Tf, f

# ---------------- 单景检测（文献法） ----------------
def detect_once(geo_file, fdi_file, sens=0.85, want_dozier=False, print_stats=False):
    with h5py.File(geo_file, "r") as g:
        sza   = read_any(g, ["Navigation/NOMSunZenith","DATA/NOMSunZenith","Data/NOMSunZenith"], "NOMSunZenith")
        satz  = read_any(g, ["Navigation/NOMSatelliteZenith","DATA/NOMSatelliteZenith","Data/NOMSatelliteZenith"], "NOMSatelliteZenith")
        sglit = read_any(g, ["Navigation/NOMSunGlintAngle","DATA/NOMSunGlintAngle","Data/NOMSunGlintAngle"], "NOMSunGlintAngle", allow_none=True)
    sza_deg   = norm_angle_deg(sza,  180.0)
    satz_deg  = norm_angle_deg(satz, 180.0)
    sglit_deg = norm_angle_deg(sglit, 360.0) if sglit is not None else None

    with h5py.File(fdi_file, "r") as f:
        T4  = lut_bt(f,
                     "Data/NOMChannel07" if "Data/NOMChannel07" in f else "DATA/NOMChannel07",
                     "Calibration/CALChannel07" if "Calibration/CALChannel07" in f else "CALIBRATION/CALChannel07")
        T13 = lut_bt(f,
                     "Data/NOMChannel13" if "Data/NOMChannel13" in f else "DATA/NOMChannel13",
                     "Calibration/CALChannel13" if "Calibration/CALChannel13" in f else "CALIBRATION/CALChannel13")
        vis = None
        if "Calibration/CALIBRATION_COEF(SCALE+OFFSET)" in f and (("Data/NOMChannel02" in f) or ("DATA/NOMChannel02" in f)):
            coef = f["Calibration/CALIBRATION_COEF(SCALE+OFFSET)"][:]
            dn02 = f["Data/NOMChannel02"][:] if "Data/NOMChannel02" in f else f["DATA/NOMChannel02"][:]
            s02, o02 = float(coef[1,0]), float(coef[1,1])
            vis = np.clip(dn02.astype(np.float64) * s02 + o02, 0.0, 1.2)

    cand = contextual_candidates(T4, T13, sza_deg, satz_deg, sun_glint=sglit_deg, vis=vis, sens=sens, debug=print_stats)

    if print_stats:
        BTD = T4 - T13
        print("T4[K]  p50/p95/max =", np.nanpercentile(T4,50), np.nanpercentile(T4,95), np.nanmax(T4))
        print("BTD[K] p50/p95/max =", np.nanpercentile(BTD,50), np.nanpercentile(BTD,95), np.nanmax(BTD))
        print(f"SZA  min/50%/max  = {np.nanmin(sza_deg):.1f}, {np.nanpercentile(sza_deg,50):.1f}, {np.nanmax(sza_deg):.1f}")
        print(f"SatZen min/50%/max= {np.nanmin(satz_deg):.1f}, {np.nanpercentile(satz_deg,50):.1f}, {np.nanmax(satz_deg):.1f}")

    # 返回 T4/T13 供小时聚合与可选 Dozier
    return cand, T4, T13

# ---------------- 配对/聚合 ----------------
def find_pairs(data_dir, date_str, max_geo_gap_min=5):
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(".hdf")]
    fdis = [f for f in files if is_fdi(f) and date_str in os.path.basename(f)]
    geos = [f for f in files if is_geo(f) and date_str in os.path.basename(f)]
    if not fdis or not geos: return {}
    geo_list = []
    for g in geos:
        g0, g1 = parse_time_window(g)
        if g0 and g1: geo_list.append((g, g0, g1))
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

def plot_hour_png(lat_list, lon_list, hour_dt, png_path):
    plt.figure(figsize=(11, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.7)
    provinces = cfeature.NaturalEarthFeature('cultural','admin_1_states_provinces_lines','50m',facecolor='none')
    ax.add_feature(provinces, edgecolor='gray', linewidth=0.4)
    if lat_list:
        ax.scatter(lon_list, lat_list, s=50, c='red', edgecolors="k", linewidths=0.3, alpha=0.9,
                   transform=ccrs.PlateCarree())
    plt.title(f"FY-4B Hotspots | {hour_dt:%Y-%m-%d %H}:00 (Beijing Time)")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

# ---------------- 主流程 ----------------
def run_daily(data_dir, date_str, out_dir,
              hits_min=HITS_PERSIST_MIN, resolution=RES_DEFAULT, sub_lon=SUB_LON,
              sens=0.85, want_dozier=False, max_resid_px=1.5):
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

        cands, T4s, T13s = [], [], []
        for i, (geo, fdi, f0, f1) in enumerate(pairs):
            cand, T4, T13 = detect_once(geo, fdi, sens=sens, want_dozier=want_dozier, print_stats=(i==0))
            cands.append(cand.astype(np.uint8))
            T4s.append(T4); T13s.append(T13)

        if not cands: 
            continue

        freq = np.sum(cands, axis=0).astype(np.uint16)
        persistent = (freq >= hits_min)
        yy, xx = np.where(persistent)

        lat_list, lon_list, hits_list, maxT_list = [], [], [], []
        Tf_list, f_list = [], []

        if len(yy) > 0:
            any_hot = True
            T4_stack  = np.stack([np.where(np.isfinite(t),  t,  -np.inf) for t in T4s],  axis=0)
            T13_stack = np.stack([np.where(np.isfinite(t), t,  -np.inf) for t in T13s], axis=0)
            maxT4_at_pix  = np.max(T4_stack[:, yy, xx], axis=0)
            maxT13_at_pix = np.max(T13_stack[:, yy, xx], axis=0)

            for k in range(len(yy)):
                lin, col = float(yy[k]), float(xx[k])
                lat, lon, resid = inverter.invert_one(lin, col)
                if (resid <= max_resid_px) and (LAT_MIN <= lat <= LAT_MAX) and (LON_MIN <= lon <= LON_MAX):
                    lat_list.append(lat); lon_list.append(lon)
                    hits_list.append(int(freq[int(lin), int(col)]))
                    maxT_list.append(float(maxT4_at_pix[k]))

                    if want_dozier:
                        # 简单局地背景均值（21窗口）
                        Hh = 10
                        ys = int(max(0, lin - Hh)); ye = int(min(T4s[0].shape[0], lin + Hh + 1))
                        xs = int(max(0, col - Hh)); xe = int(min(T4s[0].shape[1], col + Hh + 1))
                        bg_t4  = np.nanmean(np.where(np.isfinite(T4s[0][ys:ye, xs:xe]),  T4s[0][ys:ye, xs:xe],  np.nan))
                        bg_t13 = np.nanmean(np.where(np.isfinite(T13s[0][ys:ye, xs:xe]), T13s[0][ys:ye, xs:xe], np.nan))
                        Tf, farea = dozier_estimate(maxT4_at_pix[k], maxT13_at_pix[k], bg_t4, bg_t13)
                        Tf_list.append(Tf); f_list.append(farea)

        # CSV
        csv_hour = os.path.join(out_dir, f"hotspots_{hour_dt:%Y%m%d_%H}00.csv")
        if lat_list:
            data = {
                "Hour": [hour_dt.strftime("%Y-%m-%d %H:00 (Beijing Time)")] * len(lat_list),
                "Latitude": lat_list, "Longitude": lon_list,
                "Hits": hits_list, "Max_BT_3p75": maxT_list
            }
            if want_dozier:
                data["Fire_T(K)"] = Tf_list if Tf_list else [np.nan]*len(lat_list)
                data["Fire_frac"] = f_list  if f_list else [np.nan]*len(lat_list)
            pd.DataFrame(data).to_csv(csv_hour, index=False)
        else:
            cols = ["Hour","Latitude","Longitude","Hits","Max_BT_3p75"]
            if want_dozier: cols += ["Fire_T(K)","Fire_frac"]
            pd.DataFrame(columns=cols).to_csv(csv_hour, index=False)

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
        print("这一天没有检测到有效热点（情境阈值/兜底后仍为空）。")

# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="FY-4B | 指定日期逐小时火点聚合（文献法·情境阈值 + 可选Dozier反演·更敏版）")
    ap.add_argument("--dir",   required=True, help="HDF 目录（同层放 GEO 与 FDI）")
    ap.add_argument("--date",  required=True, help="目标日期：2025-10-20 或 20251020")
    ap.add_argument("--out",   required=True, help="输出目录")
    ap.add_argument("--hits",  type=int, default=HITS_PERSIST_MIN, help="持久热点最少命中次数，默认1")
    ap.add_argument("--res",   default=RES_DEFAULT, choices=list(COFF.keys()), help="分辨率键，默认4000M")
    ap.add_argument("--sublon", type=float, default=SUB_LON, help="星下点经度，FY-4B=105.0")
    ap.add_argument("--sens",  type=float, default=0.85, help="灵敏度：<1 更敏，>1 更保守，默认0.85")
    ap.add_argument("--dozier", type=int, default=0, help="是否进行Dozier反演(0/1)，默认0")
    args = ap.parse_args()

    ds = args.date.replace("-", "")
    if len(ds) != 8:
        sys.exit("日期格式不对，用 2025-10-20 或 20251020")

    HITS_PERSIST_MIN = max(1, int(args.hits))
    run_daily(args.dir, ds, args.out,
              hits_min=HITS_PERSIST_MIN,
              resolution=args.res, sub_lon=args.sublon,
              sens=float(args.sens), want_dozier=bool(args.dozier))

