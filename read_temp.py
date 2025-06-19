import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

# Configuration
file_path = 'itp65cormat.nc'  # Update with your NetCDF file path
ds = nc.Dataset(file_path)
itp = 'ITP65'  # Update with your ITP name if needed

# --- NEW: read all profile IDs and extrema depths ---
profile_ids    = ds.variables['FloatID'][:]          # array of FloatID for each profile
depth_max_all  = ds.variables['depth_max_T'][:]      # depth of maximum temperature :contentReference[oaicite:0]{index=0}
depth_min_all  = ds.variables['depth_min_T'][:]      # depth of minimum temperature :contentReference[oaicite:1]{index=1}

# Print shapes (sizes) of the three arrays
print(f"profile_ids shape : {profile_ids.shape}")
print(f"depth_max_T shape : {depth_max_all.shape}")
print(f"depth_min_T shape : {depth_min_all.shape}\n")

sort_idx = np.argsort(profile_ids)
profile_ids   = profile_ids[sort_idx]
depth_max_all = depth_max_all[sort_idx]
depth_min_all = depth_min_all[sort_idx]

print("Profile_ID    depth_max_T (m)    depth_min_T (m)")
for pid, dmax, dmin in zip(profile_ids, depth_max_all, depth_min_all):
    print(f"{int(pid):>10d}    {dmax:>14.2f}    {dmin:>14.2f}")