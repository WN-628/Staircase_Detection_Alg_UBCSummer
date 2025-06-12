import gsw
import netCDF4 as netcdf
import numpy as np
import scipy.interpolate as interpolate
import scipy.stats as stats
import scipy.ndimage as ndimage
import scipy.io
import pandas as pd
import ftplib as ftp
import wget 
import os
import h5py
import re

from config import FIXED_RESOLUTION_METER

def load_data_mat_zip(path, profiles, interp=True, resolution=FIXED_RESOLUTION_METER):
    valid_profiles = []
    target_levels = np.arange(0, 2000, resolution)
    # print("Files in tmp/:", profiles)

    for fname in profiles:
        if not fname.endswith(".mat"):
            continue
        try:
            with h5py.File(os.path.join(path, fname), 'r') as f:
                pressure = f['pr_filt'][:].flatten()
                temperature = f['te_cor'][:].flatten()
                salinity = f['sa_cor'][:].flatten()
                lat = float(f['latitude'][0][0])
                lon = float(f['longitude'][0][0])
        except Exception as e:
            # print(f"Skipping {fname}: {e}")
            continue

        # print(f"{fname}: pressure min = {pressure.min()}, max = {pressure.max()}, steps = {np.mean(np.diff(pressure))}")

        if pressure.min() > 0 and pressure.max() > 500:
            p_interp = target_levels
            try:
                temp_interp = np.interp(p_interp, pressure, temperature)
                salt_interp = np.interp(p_interp, pressure, salinity)
            except Exception as e:
                # print(f"Interpolation failed for {fname}: {e}")
                continue

            sa = gsw.SA_from_SP(salt_interp, p_interp, lon, lat)
            ct = gsw.CT_from_t(sa, temp_interp, p_interp)

            profile = {
                "p": p_interp,
                "ct": ct,
                "sa": sa,
                "lat": lat,
                "lon": lon,
                "juld": 0,  # default value (not found in .mat)
                "prof_no": int(fname[-8:-4]) if fname[-8:-4].isdigit() else 0
            }
            valid_profiles.append(profile)

    N = len(valid_profiles)
    array_shape = (N, len(target_levels))

    p = np.ma.masked_all(array_shape)
    ct = np.ma.masked_all(array_shape)
    sa = np.ma.masked_all(array_shape)
    lat = np.ma.masked_all(N)
    lon = np.ma.masked_all(N)
    juld = np.ma.masked_all(N)
    prof_no = np.zeros(N, dtype=int)

    for i, prof in enumerate(valid_profiles):
        p[i, :] = prof['p']
        ct[i, :] = prof['ct']
        sa[i, :] = prof['sa']
        lat[i] = prof['lat']
        lon[i] = prof['lon']
        juld[i] = prof['juld']
        prof_no[i] = prof['prof_no']
        
    # print(f"✅ Loaded {len(valid_profiles)} valid profile(s): {[p['prof_no'] for p in valid_profiles]}")

    return prof_no, p, lat, lon, ct, sa, juld
  
  
  
def load_data_csv_zip(path, profiles, interp=True, resolution=FIXED_RESOLUTION_METER):
    valid_profiles = []
    target_levels = np.arange(0, 2000, resolution)

    for fname in profiles:
        if not fname.endswith(".csv") or os.path.basename(fname).startswith("._"):
            continue

        full_path = os.path.join(path, fname)
        try:
            df = pd.read_csv(full_path)
            df.columns = df.columns.str.strip().str.lower()
            pressure = df['depth'].to_numpy().flatten() if 'depth' in df.columns else df.iloc[:, 0].to_numpy().flatten()
            temperature = df['temperature'].to_numpy().flatten() if 'temperature' in df.columns else df.iloc[:, 1].to_numpy().flatten()
            salinity = df['salinity'].to_numpy().flatten() if 'salinity' in df.columns else df.iloc[:, 2].to_numpy().flatten()
        except Exception as e:
            print(f"❌ Failed to read {fname}: {e}")
            continue

        # Extract lat/lon from filename
        try:
            base = os.path.basename(fname)
            parts = base.split('_')
            lon = float(parts[1])
            lat = float(parts[2])
        except Exception as e:
            print(f"⚠️ Failed to extract lat/lon from {fname}: {e}")
            lat, lon = 0.0, 0.0

        if pressure.size == 0 or pressure.max() <= 500:
            print(f"⛔ Skipping {fname}: invalid pressure range")
            continue

        try:
            if interp:
                max_p = pressure.max()
                p_interp = np.arange(0, max_p + resolution, resolution)
                temp_interp = np.interp(p_interp, pressure, temperature)
                salt_interp = np.interp(p_interp, pressure, salinity)

                sa = gsw.SA_from_SP(salt_interp, p_interp, lon, lat)
                ct = gsw.CT_from_t(sa, temp_interp, p_interp)

                profile = {
                    "p": p_interp,
                    "ct": ct,
                    "sa": sa,
                }
            else:
                sa = gsw.SA_from_SP(salinity, pressure, lon, lat)
                ct = gsw.CT_from_t(sa, temperature, pressure)
                profile = {
                    "p": pressure,
                    "ct": ct,
                    "sa": sa,
                }

            match = re.search(r"cor(\d{4})_", base)
            prof_number = int(match.group(1)) if match else 0
            profile.update({
                "lat": lat,
                "lon": lon,
                "juld": 0,
                "prof_no": prof_number
            })
            valid_profiles.append(profile)
        except Exception as e:
            print(f"⚠️ GSW conversion failed for {fname}: {e}")
            continue

    N = len(valid_profiles)
    if N == 0:
        print("⛔ No valid profiles found.")
        return [], None, None, None, None, None, None
    
    N = len(valid_profiles)
    array_shape = (N, len(target_levels))

    p = np.ma.masked_all(array_shape)
    ct = np.ma.masked_all(array_shape)
    sa = np.ma.masked_all(array_shape)
    lat = np.ma.masked_all(N)
    lon = np.ma.masked_all(N)
    juld = np.ma.masked_all(N)
    prof_no = np.zeros(N, dtype=int)

    # max_len = max(len(p["p"]) for p in valid_profiles)

    # p = np.ma.masked_all((N, max_len))
    # ct = np.ma.masked_all((N, max_len))
    # sa = np.ma.masked_all((N, max_len))
    # lat = np.ma.masked_all(N)
    # lon = np.ma.masked_all(N)
    # juld = np.ma.masked_all(N)
    # prof_no = np.zeros(N, dtype=int)

    for i, prof in enumerate(valid_profiles):
        L = len(prof["p"])
        p[i, :L] = prof["p"]
        ct[i, :L] = prof["ct"]
        sa[i, :L] = prof["sa"]
        lat[i] = prof["lat"]
        lon[i] = prof["lon"]
        juld[i] = prof["juld"]
        prof_no[i] = prof["prof_no"]

    print(f"✅ Loaded {N} valid profile(s)")
    return prof_no, p, lat, lon, ct, sa, juld
