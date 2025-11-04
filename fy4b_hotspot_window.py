# -*- coding: utf-8 -*-
# fy4b_hotspot_hourly.py

import os
import re
import h5py
import numpy as np
from datetime import datetime, timedelta
from scipy.ndimage import label, binary_opening, binary_closing, generate_binary_structure, median_filter
import rasterio
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------- 几何 + 网格 ----------
EA, EB, H = 6378.137, 6356.7523, 42164.0   # km
LAMBDA_D = np.deg2rad(105.0)
COFF = {"4000M": 1373.5}; LOFF = {"4000M": 1373.5}
CFAC = {"4000M": 10233137}; LFAC = {"4000M": 10233137}
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, STEP = 18.0, 54.0, 72.0, 136.0, 0.04

# ---------- 判别阈值 ----------
DAY_SZA_MAX, NIGHT_SZA_MIN = 85.0, 88.0
ABS_T7_DAY, ABS_T7_NIGHT   = 315.0, 310.0
Z_BTD_THRESH, Z_T7_THRESH  = 3.0, 2.0
MAD_EPS, BK = 0.5, 11
VIS_REFL_CLOUD, COLD_BT13_CLOUD = 0.35, 290.0
MIN_CLUSTER_PIX = 2
GEO_MAX_GAP_MIN = 5

# ---------- 工具 ----------
def latlon2linecolumn(lat_deg, lon_deg, res="4000M"):
    lat = np.deg2rad(lat_deg).astype(np.float64)
    lon = np.deg2rad(lon_deg).astype(np.float64)
    k   = (EB**2)/(EA**2)
    phi = np.arctan(k * np.tan(lat))
    cph = np.cos(phi)
    re  = EB / np.sqrt(1.0 - (1.0 - k) * (cph**2))
    dlam= lon - LAMBDA_D
    r1 = H - re * cph * np.cos(dlam)
    r2 = -re * cph * np.sin(dlam)
    r3 =  re * np.sin(phi)
    rn = np.sqrt(r1*r1 + r2*r2 + r3*r3)
    visible = (r1 > 0)
    x = np.arctan(-r2 / r1); y = np.arcsin(-r3 / rn)
    col = COFF[res] + x * (CFAC[res] * 2.0**-16)
    row = LOFF[res] + y * (LFAC[res] * 2.0**-16)
    col_idx = np.floor(col).astype(np.int32)
    row_idx = np.floor(row).astype(np.int32)
    col_idx[~visible] = -1; row_idx[~visible] = -1
    return row_idx, col_idx

def make_grid():
    lats = np.arange(LAT_MAX, LAT_MIN - 1e-9, -STEP)
    lons = np.arange(LON_MIN, LON_MAX + 1e-9,  STEP)
    return np.meshgrid(lats, lons, indexing="ij")  # (nrow,ncol)

def read_any(h, paths, desc="dataset", allow_none=False):
    for p in paths:
        if p in h: return h[p][:]
    if allow_none: return None
    raise KeyError(f"{desc} not found. Tried: {paths}")

def get_calib_scale_offset(h, ch_idx):
    for p in ["Calibration/CALIBRATION_COEF(SCALE+OFFSET)",
              "CALIBRATION/CALIBRATION_COEF(SCALE+OFFSET)",
              "Calibration/CALIBRATION_COEF",
              "CALIBRATION/CALIBRATION_COEF"]:
        if p in h:
            coef = h[p][:]
            return float(coef[ch_idx,0]), float(coef[ch_idx,1])
    scale  = read_any(h, ["Calibration/CALIBRATION_SCALE","CALIBRATION/CALIBRATION_SCALE",
                          "Calibration/SCALE","CALIBRATION/SCALE"], "SCALE")
    offset = read_any(h, ["Calibration/CALIBRATION_OFFSET","CALIBRATION/CALIBRATION_OFFSET",
                          "Calibration/OFFSET","CALIBRATION/OFFSET"], "OFFSET")
    return float(scale[ch_idx]), float(offset)

def dn_to_lin(dn, scale, offset):
    arr = dn.astype(np.float64)
    arr = np.where(arr == 65535, np.nan, arr)
    return arr * scale + offset

def radiance_to_bt(L, w_um):
    L = np.maximum(L, 1e-10)
    c1, c2 = 1.19104e8, 14387.8
    return (c2 / w_um) / np.log(c1/(w_um**5 * L) + 1.0)

def robust_z(arr, size=BK):
    if not np.any(np.isfinite(arr)):
        return np.full_like(arr, np.nan, dtype=np.float64)
    base = np.nanmedian(arr)
    filled = np.where(np.isfinite(arr), arr, base)
    med = median_filter(filled, size=size, mode='nearest')
    mad = median_filter(np.abs(filled - med), size=size, mode='nearest')
    mad = np.where(mad < MAD_EPS, MAD_EPS, mad)
    return (filled - med) / (1.4826 * mad)

def write_geotiff(path, data, bounds, nodata=None, dtype=None):
    nrow, ncol = data.shape
    transform = from_bounds(*bounds, width=ncol, height=nrow)
    profile = {"driver":"GTiff","height":nrow,"width":ncol,"count":1,
               "dtype":(dtype or data.dtype),"crs":"EPSG:4326","transform":transform,
               "compress":"deflate","tiled":True}
    if nodata is not None: profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(profile["dtype"]), 1)

def parse_time_window(path):
    m = re.findall(r"_(\d{14})_", os.path.basename(path))
    if len(m) >= 2:
        t0 = datetime.strptime(m[-2], "%Y%m%d%H%M%S")
        t1 = datetime.strptime(m[-1], "%Y%m%d%H%M%S")
        return t0, t1
    return None, None

def is_fdi(p): return "_FDI-" in os.path.basename(p)
def is_geo(p): return "_GEO-" in os.path.basename(p)

def parse_cli_time(s):
    # 支持 2025-09-04T01:00 / 202509040100 / 2025-09-04 01:00
    for fmt in ("%Y-%m-%dT%H:%M","%Y%m%d%H%M","%Y-%m-%d %H:%M"):
        try: return datetime.strptime(s, fmt)
        except: pass
    raise ValueError(f"无法解析时间: {s}")

# ---------- 单时次检测（复用之前逻辑） ----------
def detect_hotspots_once(fdi_file, geo_file, out_dir, LAT, LON, rows, cols):
    t0, t1 = parse_time_window(fdi_file)
    tag = (t0.strftime("%Y%m%d_%H%M%S") if t0 else "unknown")
    bounds = (LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)

    with h5py.File(fdi_file, "r") as fd:
        ch07 = read_any(fd, ["Data/NOMChannel07","DATA/NOMChannel07"], "Ch07")
        ch13 = read_any(fd, ["Data/NOMChannel13","DATA/NOMChannel13"], "Ch13")
        s07,o07 = get_calib_scale_offset(fd, 6)
        s13,o13 = get_calib_scale_offset(fd, 12)
        vis = read_any(fd, ["Data/NOMChannel03","DATA/NOMChannel03",
                            "Data/NOMChannel02","DATA/NOMChannel02"], "VIS", allow_none=True)
        vis_coef = None
        if vis is not None:
            vis_idx = 2 if "NOMChannel03" in fd["Data"].keys() else 1
            vis_coef = get_calib_scale_offset(fd, vis_idx)

    Hh, Ww = ch07.shape
    ok = (rows>=0) & (cols>=0) & (rows<Hh) & (cols<Ww)

    dn07 = np.full(LAT.shape, np.nan, dtype=np.float32)
    dn13 = np.full(LAT.shape, np.nan, dtype=np.float32)
    dn07[ok] = ch07[rows[ok], cols[ok]]
    dn13[ok] = ch13[rows[ok], cols[ok]]

    T7  = radiance_to_bt(dn_to_lin(dn07, s07, o07),  3.75)
    T13 = radiance_to_bt(dn_to_lin(dn13, s13, o13), 10.8)
    BTD = T7 - T13

    vis_ref = None
    if vis is not None and vis_coef is not None:
        dnvis = np.full(LAT.shape, np.nan, dtype=np.float32)
        dnvis[ok] = vis[rows[ok], cols[ok]]
        sV,oV = vis_coef
        vis_ref = dn_to_lin(dnvis, sV, oV)

    with h5py.File(geo_file, "r") as fg:
        sza = read_any(fg, ["Navigation/NOMSunZenith","NAVIGATION/NOMSunZenith",
                            "/Navigation/NOMSunZenith"], "SunZenith")
    sza_grid = np.full(LAT.shape, np.nan, dtype=np.float32)
    sza_grid[ok] = sza[rows[ok], cols[ok]]

    z_btd = robust_z(BTD, size=BK)
    z_t7  = robust_z(T7,  size=BK)
    day   = (sza_grid < DAY_SZA_MAX)
    night = (sza_grid > NIGHT_SZA_MIN)

    cloud_day = np.zeros_like(day, dtype=bool)
    if vis_ref is not None:
        cloud_day = (vis_ref > VIS_REFL_CLOUD) & (T13 < COLD_BT13_CLOUD)

    cand_day =  day   & (z_btd > Z_BTD_THRESH) & (z_t7 > Z_T7_THRESH) & (T7 > ABS_T7_DAY)   & ~cloud_day
    cand_ngt = night & (z_btd > Z_BTD_THRESH)                                          & (T7 > ABS_T7_NIGHT)
    cand = cand_day | cand_ngt
    cand &= np.isfinite(T7) & np.isfinite(T13) & np.isfinite(BTD)

    st = generate_binary_structure(2,1)
    cand = binary_opening(cand, st, 1)
    cand = binary_closing(cand, st, 1)
    lab, nlab = label(cand)
    if nlab:
        sizes = np.bincount(lab.ravel())
        cand[np.isin(lab, np.where(sizes < MIN_CLUSTER_PIX)[0])] = False

    # 输出每时次
    subdir = os.path.join(out_dir, tag); os.makedirs(subdir, exist_ok=True)
    import pandas as pd
    yy, xx = np.where(cand)
    csvp = os.path.join(subdir, f"hotspots_{tag}.csv")
    if len(yy) > 0:
        pd.DataFrame({
            "Latitude":  LAT[yy, xx],
            "Longitude": LON[yy, xx],
            "BT_3p75":   T7[yy, xx],
            "BT_10p8":   T13[yy, xx],
            "BTD":       BTD[yy, xx],
            "Z_BTD":     z_btd[yy, xx],
            "Z_T7":      z_t7[yy, xx],
            "SunZenith": sza_grid[yy, xx],
            "IsDay":     day[yy, xx].astype(int)
        }).to_csv(csvp, index=False)
    else:
        open(csvp, "w").write("Latitude,Longitude,BT_3p75,BT_10p8,BTD,Z_BTD,Z_T7,SunZenith,IsDay\n")

    write_geotiff(os.path.join(subdir, f"btdz_{tag}.tif"),
                  np.where(np.isfinite(z_btd), z_btd, -9999).astype(np.float32),
                  (LON_MIN, LAT_MIN, LON_MAX, LAT_MAX), nodata=-9999, dtype="float32")
    write_geotiff(os.path.join(subdir, f"hotmask_{tag}.tif"),
                  cand.astype(np.uint8),
                  (LON_MIN, LAT_MIN, LON_MAX, LAT_MAX), nodata=0, dtype="uint8")

    # 预览
    fig = plt.figure(figsize=(11,7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    im = ax.imshow(T7, extent=(LON_MIN,LON_MAX,LAT_MIN,LAT_MAX),
                   origin='lower', cmap='inferno', vmin=280, vmax=330)
    ax.set_extent((LON_MIN,LON_MAX,LAT_MIN,LAT_MAX)); ax.coastlines('50m', linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.7)
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    plt.colorbar(im, ax=ax, label='Brightness Temperature (K)')
    ax.set_title(f'FY-4B 3.75µm BT | hotspots={len(yy)} | {t0}~{t1}')
    plt.tight_layout(); fig.savefig(os.path.join(subdir, f"bt7_preview_{tag}.png"), dpi=200); plt.close(fig)

    return cand, T7, tag  # cand用于窗口聚合

# ---------- 时间窗处理 ----------
def pair_in_window(data_dir, t_start, t_end):
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(".hdf")]
    fdis  = [f for f in files if is_fdi(f)]
    geos  = [f for f in files if is_geo(f)]

    # GEO索引：结束时间
    geo_list = []
    for g in geos:
        g0,g1 = parse_time_window(g)
        if g0 and g1: geo_list.append((g,g0,g1))
    geo_list.sort(key=lambda x: x[2])

    # 选取窗口内的FDI，并找最近前置GEO（结束≤FDI开始，gap≤5min）
    pairs = []
    for f in fdis:
        f0,f1 = parse_time_window(f)
        if not (f0 and f1): continue
        if not (f0 >= t_start and f0 < t_end):  # 按 FDI 开始时刻落入窗口
            continue
        best, best_gap = None, timedelta(hours=999)
        for g,g0,g1 in geo_list:
            if g1 <= f0:
                gap = f0 - g1
                if gap < best_gap: best, best_gap = g, gap
        if best and best_gap <= timedelta(minutes=GEO_MAX_GAP_MIN):
            pairs.append((f, best))
    return pairs

def process_window(data_dir, out_dir, t_start, t_end, k_required=2):
    os.makedirs(out_dir, exist_ok=True)
    LAT, LON = make_grid()
    rows, cols = latlon2linecolumn(LAT, LON, "4000M")

    pairs = pair_in_window(data_dir, t_start, t_end)
    if not pairs:
        print("时间窗内没有匹配到 FDI/GEO。"); return

    print(f"窗口 {t_start} ~ {t_end} | 匹配到 {len(pairs)} 对")
    nrow, ncol = LAT.shape
    freq = np.zeros((nrow, ncol), dtype=np.uint16)

    # 聚合元数据：每格统计最大T7、首次/末次命中时间
    first_hit = np.full((nrow, ncol), None, dtype=object)
    last_hit  = np.full((nrow, ncol), None, dtype=object)
    max_t7    = np.full((nrow, ncol), -np.inf, dtype=np.float32)

    for i,(fdi,geo) in enumerate(pairs,1):
        print(f"[{i}/{len(pairs)}] {os.path.basename(fdi)}")
        cand, T7, tag = detect_hotspots_once(fdi, geo, out_dir, LAT, LON, rows, cols)
        freq += cand.astype(np.uint16)
        # 元数据更新
        ts = datetime.strptime(tag, "%Y%m%d_%H%M%S")
        yy, xx = np.where(cand)
        for y,x in zip(yy,xx):
            if first_hit[y,x] is None or ts < first_hit[y,x]: first_hit[y,x] = ts
            if last_hit[y,x]  is None or ts > last_hit[y,x]:  last_hit[y,x]  = ts
            if T7[y,x] > max_t7[y,x]: max_t7[y,x] = T7[y,x]

    # 真火：≥k 次
    persistent = (freq >= k_required)

    # 输出窗口级产品
    bounds = (LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    write_geotiff(os.path.join(out_dir, "window_freq.tif"),
                  freq.astype(np.uint16), bounds, nodata=0, dtype="uint16")
    write_geotiff(os.path.join(out_dir, "window_persistent_mask.tif"),
                  persistent.astype(np.uint8), bounds, nodata=0, dtype="uint8")

    # 点清单
    import pandas as pd
    yy, xx = np.where(persistent)
    if len(yy) > 0:
        df = pd.DataFrame({
            "Latitude":  LAT[yy, xx],
            "Longitude": LON[yy, xx],
            "Hits":      freq[yy, xx],
            "Max_BT_3p75": max_t7[yy, xx],
            "FirstHitUTC": [first_hit[y,x].strftime("%Y-%m-%d %H:%M:%S") if first_hit[y,x] else "" for y,x in zip(yy,xx)],
            "LastHitUTC":  [last_hit[y,x].strftime("%Y-%m-%d %H:%M:%S")  if last_hit[y,x]  else "" for y,x in zip(yy,xx)]
        })
        df.to_csv(os.path.join(out_dir, "window_persistent_points.csv"), index=False)
    else:
        open(os.path.join(out_dir, "window_persistent_points.csv"),"w")\
            .write("Latitude,Longitude,Hits,Max_BT_3p75,FirstHitUTC,LastHitUTC\n")

    # 预览：频次热力
    fig = plt.figure(figsize=(11,7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    im = ax.imshow(freq, extent=(LON_MIN,LON_MAX,LAT_MIN,LAT_MAX),
                   origin='lower', cmap='hot')
    ax.set_extent((LON_MIN,LON_MAX,LAT_MIN,LAT_MAX))
    ax.coastlines('50m', linewidth=0.7); ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.7)
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    plt.colorbar(im, ax=ax, label='Detections within window')
    ax.set_title(f"FY-4B Hotspots Frequency | {t_start} ~ {t_end} | K>={k_required}")
    plt.tight_layout(); fig.savefig(os.path.join(out_dir, "window_preview.png"), dpi=200); plt.close(fig)
    print(f"完成：真火像元 {persistent.sum()} 个，频次栅格已写出。")

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="FY-4B Hotspot in Time Window (EPSG:4326)")
    ap.add_argument("--dir",  required=True, help="含 HDF 的目录")
    ap.add_argument("--out",  required=True, help="输出目录")
    ap.add_argument("--start", required=True, help="窗口开始 (e.g., 2025-09-04T01:00)")
    ap.add_argument("--end",   required=True, help="窗口结束 (e.g., 2025-09-04T02:00)")
    ap.add_argument("--k", type=int, default=2, help="同一格最少命中次数才算真火 (默认2)")
    args = ap.parse_args()

    if not os.path.isdir(args.dir): sys.exit(f"目录不存在: {args.dir}")
    os.makedirs(args.out, exist_ok=True)
    t_start = parse_cli_time(args.start); t_end = parse_cli_time(args.end)
    if not (t_end > t_start): sys.exit("时间窗无效：end 必须晚于 start")
    process_window(args.dir, args.out, t_start, t_end, k_required=args.k)
