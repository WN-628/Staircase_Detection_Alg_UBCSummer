#!/usr/bin/env python3
import netCDF4 as nc
import numpy as np
import pandas as pd

# Script to extract the profile for FloatID = 1 from 'itp65cormat.nc'
# and check for NaN values in the pressure array.

ncfile = 'itp65cormat.nc'
ds = nc.Dataset(ncfile, 'r')

# Read FloatID array and locate the index for ID=1
float_ids = ds.variables['FloatID'][:]
fid = 2
indices = np.where(float_ids == fid)[0]
if indices.size == 0:
    print(f"⚠️ FloatID {fid} not found in {ncfile}.")
    print("📋 Available FloatIDs:", sorted(float_ids.tolist()))
    ds.close()
    import sys; sys.exit(1)
idx = int(indices[0])

# Extract vlen arrays for FloatID=1
p_profile  = ds.variables['pressure'][idx]
ct_profile = ds.variables['ct'][idx]

# Salinity if available
sa_profile = None
if 'sa' in ds.variables:
    sa_profile = ds.variables['sa'][idx]

# Check for NaNs in pressure profile
if np.isnan(p_profile).any():
    print(f"⚠️ Warning: NaN values found in pressure for FloatID {fid}")
else:
    print(f"✅ No NaN values in pressure for FloatID {fid}")

# Build DataFrame for output
data = {
    'pressure_m': p_profile,
    'ct_degC':    ct_profile
}
if sa_profile is not None:
    data['sa_g_per_kg'] = sa_profile

df = pd.DataFrame(data)

# Write single CSV for FloatID=1
df.to_csv(f'ct_profile_{fid}.csv', index=False)
print(f"Wrote {len(df)} levels for FloatID {fid} → ct_profile_{fid}.csv")

ds.close()
