import os
import shutil
import zipfile
import warnings

import numpy as np
# import netCDF4 as netcdf

from data_preparation import load_data_csv_zip
from create_netcdf import create_netcdf
from staircase_detector import get_mixed_layers
from config import FIXED_RESOLUTION_METER
from after_detector import *
from check_sharpness import check_sharpness

"""
Script to detect thermohaline staircases in Ice Tethered Profiler data.
Reads .zip files from a specified input directory and writes .nc files to a specified output directory.
"""

# --- Configuration: input and output folders ---
INPUT_DIR = 'gridData_zip'
OUTPUT_DIR = 'prod_files'

# Ensure input exists
if not os.path.isdir(INPUT_DIR):
    print(f"❌ Input directory '{INPUT_DIR}' does not exist.")
    exit(1)

# Clean or create output directory
if os.path.isdir(OUTPUT_DIR):
    # List files to be deleted
    files_to_delete = []
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for name in files:
            files_to_delete.append(os.path.join(root, name))
    if files_to_delete:
        print(f"Deleting {len(files_to_delete)} files from folder '{OUTPUT_DIR}':")
        for fpath in files_to_delete:
            print(f" - {fpath}")
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('Ice tethered profiles')

# Find all zip files in the input directory
zip_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.zip')]
if not zip_files:
    print(f"⚠️ No .zip files found in '{INPUT_DIR}'")
    exit(0)

# Initialize counters for summary
processed_zip_count = 0
total_valid_profiles = 0

# Detection thresholds
thres_ml_upper  = 0.002  # mixed layer gradient threshold
thres_int_lower = 0.005  # interface gradient threshold
ml_min_length   = 0.75   # mixed layer min depth length (m)
int_min_temp    = 0.01   # interface min temperature width (°C)
cl_length       = 1.0    # connecting layer max length (m)
smooth_length   = 7      # smoothing window (grid points)

for zip_name in zip_files:
    base = os.path.splitext(zip_name)[0]
    src_zip = os.path.join(INPUT_DIR, zip_name)
    out_ncfile = os.path.join(OUTPUT_DIR, f"{base}.nc")
    print(f"📦 Processing {src_zip} → will save to {out_ncfile}")

    # 1) Extract CSVs
    tmp_dir = 'tmp'
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    with zipfile.ZipFile(src_zip, 'r') as z:
        z.extractall(tmp_dir)
    # Optionally remove the original zip to clean up
    # os.remove(src_zip)

    # 2) Gather profile files
    profiles = []
    for root, _, files in os.walk(tmp_dir):
        for f in files:
            if f.endswith('.csv') and not f.startswith('._'):
                profiles.append(os.path.join(root, f))

    # 3) Load raw profiles (no interpolation)
    prof_no, p_raw, lat, lon, ct_raw, sa_raw, dates = load_data_csv_zip(
        '', profiles, interp=False,
        resolution=FIXED_RESOLUTION_METER
    )
    N = len(prof_no)
    if N == 0:
        print(f"⚠️ No valid profiles in '{zip_name}'")
        shutil.rmtree(tmp_dir)
        continue

    # Update summary counters
    processed_zip_count += 1
    total_valid_profiles += N

    # 4) Determine maximum true profile length
    valid_mask = ~np.ma.getmaskarray(p_raw)
    lengths = valid_mask.sum(axis=1)
    max_len = int(np.max(lengths))

    # 5) Allocate per-profile grids
    p   = np.ma.masked_all((N, max_len))
    ct  = np.ma.masked_all((N, max_len))
    sa  = np.ma.masked_all((N, max_len))

    for i in range(N):
        valid = valid_mask[i]
        L = valid.sum()
        p[i, :L]  = p_raw[i, valid]
        ct[i, :L] = ct_raw[i, valid]
        sa[i, :L] = sa_raw[i, valid]

    # Clean up temp files
    shutil.rmtree(tmp_dir)

    # 5) Create NetCDF file (vlen types expected)
    fh = create_netcdf(out_ncfile, _nlevels_unused=None)

    # 6) Run staircase detection
    masks, depth_min_T, depth_max_T = get_mixed_layers(
        np.ma.copy(p), np.ma.copy(ct),
        thres_ml_upper, thres_int_lower,
        ml_min_length, int_min_temp,
        cl_length, smooth_length
    )
    cl_mushy, cl_supermushy = check_sharpness(masks.cl, cl_length)

    # 7) Write scalar metadata and outputs
    fh.variables['lat'][:]      = lat
    fh.variables['lon'][:]      = lon
    fh.variables['dates'][:]    = dates
    fh.variables['FloatID'][:]  = prof_no

    # fh.variables['cl_mushy'][:]      = cl_mushy
    # fh.variables['cl_supermushy'][:] = cl_supermushy
    fh.variables['depth_max_T'][:]   = depth_max_T
    fh.variables['depth_min_T'][:]   = depth_min_T

    # 8) Write variable-length profiles and masks
    p_var     = fh.variables['pressure']
    ct_var    = fh.variables['ct']
    sa_var    = fh.variables['sa']
    ml_var    = fh.variables['mask_ml']
    int_var   = fh.variables['mask_int']
    cl_var    = fh.variables['mask_cl']
    sc_var    = fh.variables['mask_sc']
    clm       = fh.variables['cl_mushy']
    clsm      = fh.variables['cl_supermushy']
    
    valid_mask = ~np.ma.getmaskarray(p)
    for i in range(N):
        vm = valid_mask[i]
        p_var[i]   = p[i, vm].data.astype(np.float32)
        ct_var[i]  = ct[i, vm].data
        sa_var[i]  = sa[i, vm].data
        ml_var[i]  = masks.ml[i, vm]
        int_var[i] = masks.int[i, vm]
        cl_var[i]  = masks.cl[i, vm]
        sc_var[i]  = masks.sc[i, vm]
        clm[i]     = cl_mushy[i, vm]
        clsm[i]    = cl_supermushy[i, vm]

    fh.close()
    print(f"✅ Processed {base}: {N} profiles → {out_ncfile}")

# Final summary
print(f"✅ Processing complete. NC files are in '{OUTPUT_DIR}'.")
print(f"Processed {processed_zip_count} zip files with a total of {total_valid_profiles} valid profiles.")

# --- Post-run check on valid pressure data ---
# Only inspect unmasked entries to ensure no NaN
try:
    valid_pressures = p.data[~p.mask]
    if np.isnan(valid_pressures).any():
        print("⚠️ Warning: NaN found in valid pressure data of 'p'.")
    else:
        print("✅ No NaN in valid pressure data of 'p'.")
except NameError:
    pass
