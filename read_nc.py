import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

# ── USER CONFIG ─────────────────────────────────────────
nc_path   = 'prod_files/itp18cormat.nc'
target_id = 281
# ────────────────────────────────────────────────────────

# 1) open and find the profile index
ds        = nc.Dataset(nc_path, 'r')
float_ids = ds.variables['FloatID'][:]                # shape (Nobs,)
inds      = np.where(float_ids == target_id)[0]
if inds.size == 0:
    raise ValueError(f"No profile with FloatID={target_id}")
prof = int(inds[0])

# 2) helper to grab a vlen variable for that profile
def grab(varname):
    v = ds.variables[varname]
    arr = v[prof]            # returns a true 1-D numpy array for vlen vars
    # if it somehow comes back masked, fill with NaN
    if isinstance(arr, np.ma.MaskedArray):
        arr = arr.filled(np.nan)
    return np.array(arr)

# 3) read your six series
pressure      = grab('pressure')         # e.g. dtype float32
ct            = grab('ct')               # dtype float64
mask_ml       = grab('mask_ml').astype(bool)
mask_int      = grab('mask_int').astype(bool)
cl_mushy      = grab('cl_mushy').astype(bool)
cl_supermushy = grab('cl_supermushy').astype(bool)

ds.close()

# 4) plot as before
plt.figure(figsize=(6, 8))
plt.plot(ct, pressure, '-', label='CT profile', linewidth=1.5)

for mk, marker, lbl in [
    (mask_ml,       'o', 'Mixed Layer'),
    (mask_int,      'o', 'Interface'),
    (cl_mushy,      'x', 'Mushy'),
    (cl_supermushy, '^', 'Supermushy'),
]:
    plt.scatter(ct[mk], pressure[mk], marker=marker, label=lbl)

plt.gca().invert_yaxis()
plt.xlabel('Conservative Temperature (°C)')
plt.ylabel('Depth (m)')
plt.title(f'Profile FloatID {target_id}')
plt.legend(loc='best', fontsize='small')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
