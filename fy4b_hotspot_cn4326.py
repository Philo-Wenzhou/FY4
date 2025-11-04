# fy4b_hotspot_cn4326.py
import os
import re
import h5py
import numpy as np
from scipy.ndimage import median_filter, label
import rasterio
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# -------------------- 常量（几何 + 网格） --------------------
EA = 6378.137      # km
EB = 6356.7523     # km
H  = 42164.0       # km
LAMBDA_D = np.deg2rad(105.0)

COFF = {"4000M": 1373.5}
LOFF = {"4000M": 1373.5}
CFAC = {"4000M": 10233137}
LFAC = {"4000M": 10233137}

LAT_MIN, LAT_MAX = 18.0, 54.0
LON_MIN, LON_MAX = 72.0, 136.0
STEP = 0.04  # deg

# ---- 判别阈值 ----
DAY_SZA_MAX, NIGHT_SZA_MIN = 85.0, 88.0
ABS_T7_DAY, ABS_T7_NIGHT = 315.0, 310.0
Z_BTD_THRESH, Z_T7_THRESH = 3.0, 2.0
MAD_EPS, BK = 0.5, 11
VIS_REFL_CLOUD, COLD_BT13_CLOUD = 0.35, 290.0
MIN_CLUSTER_PIX = 2

# -------------------- 工具函数 --------------------
def latlon2linecolumn(lat_deg, lon_deg, resolution="4000M"):
    """经纬度(°)->行列索引(0基)，不可见像元=-1"""
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

    x = np.arctan(-r2 / r1)       # 弧度
    y = np.arcsin(-r3 / rn)       # 弧度

    col = COFF[resolution] + x * (CFAC[resolution] * 2.0**-16)
    row = LOFF[resolution] + y * (LFAC[resolution] * 2.0**-16)

    col_idx = np.floor(col).astype(np.int32)
    row_idx = np.floor(row).astype(np.int32)
    col_idx[~visible] = -1
    row_idx[~visible] = -1
    return row_idx, col_idx

def make_grid(lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX, step=STEP):
    lats = np.arange(lat_max, lat_min - 1e-9, -step)  # 北->南
    lons = np.arange(lon_min, lon_max + 1e-9,  step)  # 西->东
    LAT, LON = np.meshgrid(lats, lons, indexing="ij") # (nrow,ncol)
    return LAT, LON

def read_any(h, candidates, desc="dataset", allow_none=False):
    for p in candidates:
        try:
            return h[p][:]
        except KeyError:
            continue
    if allow_none:
        return None
    raise KeyError(f"{desc} not found. Tried: {candidates}")

def get_calib_scale_offset(h, ch_idx):
    # 合表
    for p in [
        "Calibration/CALIBRATION_COEF(SCALE+OFFSET)",
        "CALIBRATION/CALIBRATION_COEF(SCALE+OFFSET)",
        "Calibration/CALIBRATION_COEF",
        "CALIBRATION/CALIBRATION_COEF",
    ]:
        if p in h:
            coef = h[p][:]
            return float(coef[ch_idx,0]), float(coef[ch_idx,1])
    # 分表
    scale = read_any(h, ["Calibration/CALIBRATION_SCALE","CALIBRATION/CALIBRATION_SCALE",
                         "Calibration/SCALE","CALIBRATION/SCALE"], "SCALE")
    offset= read_any(h, ["Calibration/CALIBRATION_OFFSET","CALIBRATION/CALIBRATION_OFFSET",
                         "Calibration/OFFSET","CALIBRATION/OFFSET"], "OFFSET")
    return float(scale[ch_idx]), float(offset[ch_idx])

def dn_to_lin(dn, scale, offset):
    arr = dn.astype(np.float64)
    arr = np.where(arr==65535, np.nan, arr)
    return arr * scale + offset

def radiance_to_bt(L, wavelength_um):
    L = np.maximum(L, 1e-10)
    c1, c2 = 1.19104e8, 14387.8
    return (c2 / wavelength_um) / np.log(c1/(wavelength_um**5 * L) + 1.0)

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
    """bounds=(left,bottom,right,top) in EPSG:4326"""
    nrow, ncol = data.shape
    transform = from_bounds(*bounds, width=ncol, height=nrow)
    profile = {
        "driver": "GTiff", "height": nrow, "width": ncol,
        "count": 1, "dtype": (dtype or data.dtype),
        "crs": "EPSG:4326", "transform": transform,
        "compress": "deflate", "tiled": True
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(profile["dtype"]), 1)

def parse_ts_tag(path):
    m = re.search(r"_(\d{14})_", os.path.basename(path))
    return (m.group(1) if m else "unknown")

# -------------------- 主流程（单时次） --------------------
def detect_hotspots(fdi_file, geo_file, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    tag = parse_ts_tag(fdi_file)

    # 规则网格
    LAT, LON = make_grid()
    rows, cols = latlon2linecolumn(LAT, LON, "4000M")

    # 读 FDI
    with h5py.File(fdi_file, "r") as fd:
        ch07 = read_any(fd, ["Data/NOMChannel07","DATA/NOMChannel07"], "Ch07")
        ch13 = read_any(fd, ["Data/NOMChannel13","DATA/NOMChannel13"], "Ch13")
        coef = get_calib_scale_offset(fd, 6), get_calib_scale_offset(fd, 12)

        # 可见光（可选，03优先，次选02）
        vis = read_any(fd, ["Data/NOMChannel03","DATA/NOMChannel03",
                            "Data/NOMChannel02","DATA/NOMChannel02"], "VIS", allow_none=True)
        vis_coef = None
        if vis is not None:
            vis_idx = 2 if ("NOMChannel03" in [k for k in fd["Data"].keys()]) else 1
            vis_coef = get_calib_scale_offset(fd, vis_idx)

    Hh, Ww = ch07.shape
    ok = (rows>=0) & (cols>=0) & (rows<Hh) & (cols<Ww)

    dn07 = np.full(LAT.shape, np.nan, dtype=np.float32)
    dn13 = np.full(LAT.shape, np.nan, dtype=np.float32)
    dn07[ok] = ch07[rows[ok], cols[ok]]
    dn13[ok] = ch13[rows[ok], cols[ok]]

    s07,o07 = coef[0]
    s13,o13 = coef[1]
    L7  = dn_to_lin(dn07, s07, o07)
    L13 = dn_to_lin(dn13, s13, o13)
    T7  = radiance_to_bt(L7,  3.75)
    T13 = radiance_to_bt(L13, 10.8)
    BTD = T7 - T13

    vis_ref = None
    if vis is not None and vis_coef is not None:
        dnvis = np.full(LAT.shape, np.nan, dtype=np.float32)
        dnvis[ok] = vis[rows[ok], cols[ok]]
        sV, oV = vis_coef
        vis_ref = dn_to_lin(dnvis, sV, oV)

    # 读 SZA（GEO）
    with h5py.File(geo_file, "r") as fg:
        sza = read_any(fg, ["Navigation/NOMSunZenith","NAVIGATION/NOMSunZenith",
                            "/Navigation/NOMSunZenith"], "SunZenith")
    sza_grid = np.full(LAT.shape, np.nan, dtype=np.float32)
    sza_grid[ok] = sza[rows[ok], cols[ok]]

    # 背景 Z
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

    # 连通域去椒盐
    lab, nlab = label(cand)
    if nlab:
        sizes = np.bincount(lab.ravel())
        cand[np.isin(lab, np.where(sizes < MIN_CLUSTER_PIX)[0])] = False

    # 导出 CSV
    yy, xx = np.where(cand)
    out_csv = os.path.join(out_dir, f"hotspots_{tag}.csv")
    if len(yy) > 0:
        import pandas as pd
        df = pd.DataFrame({
            "Latitude":  LAT[yy, xx],
            "Longitude": LON[yy, xx],
            "BT_3p75":   T7[yy, xx],
            "BT_10p8":   T13[yy, xx],
            "BTD":       BTD[yy, xx],
            "Z_BTD":     z_btd[yy, xx],
            "Z_T7":      z_t7[yy, xx],
            "SunZenith": sza_grid[yy, xx],
            "IsDay":     day[yy, xx].astype(int)
        })
        df.to_csv(out_csv, index=False)
    else:
        open(out_csv, "w").write("Latitude,Longitude,BT_3p75,BT_10p8,BTD,Z_BTD,Z_T7,SunZenith,IsDay\n")

    # GeoTIFF：Z分数与掩膜（EPSG:4326）
    bounds = (LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    write_geotiff(os.path.join(out_dir, f"btdz_{tag}.tif"),
                  np.where(np.isfinite(z_btd), z_btd, -9999).astype(np.float32),
                  bounds, nodata=-9999, dtype="float32")
    write_geotiff(os.path.join(out_dir, f"hotmask_{tag}.tif"),
                  cand.astype(np.uint8), bounds, nodata=0, dtype="uint8")

    # 可选预览图
    fig = plt.figure(figsize=(11,7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    im = ax.imshow(T7, extent=bounds, origin='lower', cmap='inferno', vmin=280, vmax=330)
    ax.set_extent(bounds); ax.coastlines('50m', linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.7)
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    plt.colorbar(im, ax=ax, label='Brightness Temperature (K)')
    ax.set_title(f'FY-4B 3.75µm BT  |  hotspots={len(yy)}')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"bt7_preview_{tag}.png"), dpi=200)
    plt.close(fig)

    print(f"OK | hotspots={len(yy)} -> {out_csv}")
    return out_csv

# -------------------- 命令行入口 --------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='FY-4B 异常热点检测工具 (EPSG:4326)')
    parser.add_argument('--geo', help='GEO数据文件路径')
    parser.add_argument('--l1', '--fdi', help='FDI/L1数据文件路径')
    parser.add_argument('pos_args', nargs='*', help='位置参数: [FDI_HDF] [GEO_HDF] [OUT_DIR]')
    
    args = parser.parse_args()
    
    # 处理位置参数和命名参数
    fdi_file = None
    geo_file = None
    out_dir = None
    
    # 先检查位置参数
    if len(args.pos_args) >= 3:
        fdi_file = args.pos_args[0]
        geo_file = args.pos_args[1]
        out_dir = args.pos_args[2]
    elif len(args.pos_args) >= 2:
        fdi_file = args.pos_args[0]
        geo_file = args.pos_args[1]
    elif len(args.pos_args) == 1:
        # 特殊情况：可能是第三个参数（输出目录）
        out_dir = args.pos_args[0]
    
    # 命名参数优先级更高
    if args.l1:
        fdi_file = args.l1
    if args.geo:
        geo_file = args.geo
    
    # 验证参数
    if not fdi_file:
        parser.error('请提供FDI/L1数据文件路径 (通过位置参数或--l1/--fdi参数)')
    if not geo_file:
        parser.error('请提供GEO数据文件路径 (通过位置参数或--geo参数)')
    if not out_dir:
        out_dir = 'output_hotspots'
    
    # 检查文件是否存在
    if not os.path.exists(fdi_file):
        raise SystemExit(f"FDI文件不存在: {fdi_file}")
    if not os.path.exists(geo_file):
        raise SystemExit(f"GEO文件不存在: {geo_file}")
    
    # 创建输出目录
    os.makedirs(out_dir, exist_ok=True)
    print(f"========== FY-4B 异常热点检测 ==========")
    print(f"FDI文件: {fdi_file}")
    print(f"GEO文件: {geo_file}")
    print(f"输出目录: {out_dir}")
    
    # 执行热点检测
    detect_hotspots(fdi_file, geo_file, out_dir)
