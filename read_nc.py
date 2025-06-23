import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

# Configuration
file_path = 'prod_files/itp65cormat.nc'  # Update with your NetCDF file path
ds = nc.Dataset(file_path)
itp = 'ITP18'  # Update with your ITP name if needed

# Print dimensions
print("=== Dimensions ===")
for dim_name, dim in ds.dimensions.items():
    print(f"{dim_name}: {len(dim)}")

# Print variables and their shapes
print("\n=== Variables ===")
for var_name, var in ds.variables.items():
    print(f"{var_name}: shape = {var.shape}, dtype = {var.dtype}")

prof_no = 79              # Profile index to plot

profile_ids = ds.variables['FloatID'][:]  # 1D array of profile IDs
if prof_no in profile_ids:
    prof_idx = int(np.where(profile_ids == prof_no)[0][0])
else:
    raise ValueError(f"Profile ID {prof_no} not found; available IDs: {profile_ids}")

mask_ml_all = ds.variables['mask_ml'][:].astype(bool)

# Determine which profiles have any mixed-layer points
profiles_with_ml = profile_ids[np.any(mask_ml_all, axis=1)]
print(f"Profiles with mixed layer detected: {sorted(profiles_with_ml.tolist())}")

# Dates
time_var  = ds.variables['dates']  # dates in create_netcdf.py are Gregorian
times_all = nc.num2date(time_var[:], units=time_var.units, calendar='gregorian')

# Load dataset
pressure = ds.variables['pressure'][prof_idx, :].filled(np.nan)
ct = ds.variables['ct'][prof_idx, :]
ct_full = ds.variables['ct'][prof_idx, :].filled(np.nan)
mask_ml = ds.variables['mask_ml'][prof_idx, :].astype(bool)
mask_int = ds.variables['mask_int'][prof_idx, :].astype(bool)
mask_cl = ds.variables['mask_cl'][prof_idx, :].astype(bool)    

cl_mushy = ds.variables['cl_mushy'][prof_idx, :].astype(bool)
cl_supermushy = ds.variables['cl_supermushy'][prof_idx, :].astype(bool)

dmax = ds.variables['depth_max_T'][prof_idx]
dmin = ds.variables['depth_min_T'][prof_idx]

print("maximum pressure is:", np.max(ct))

# Mask invalid temperature
# ct = np.ma.masked_invalid(ct_full)

# Basic assertions 
assert np.any(mask_ml), "No mixed layer mask found in the data."
# assert np.any(mask_int), "No interface mask found in the data."

# # 1. get the two extreme depths for this profile


# # 2. interpolate CT at those exact depths
# #    (assumes pressure and ct_raw are 1D arrays, and pressure is monotonic)
# ct_at_dmax = np.interp(dmax, pressure, ct)
# ct_at_dmin = np.interp(dmin, pressure, ct)

# Get profile date string
timestamp = times_all[prof_idx]
date_str  = timestamp.strftime('%Y-%m-%d')

# Plot temperature profile
plt.figure(figsize=(6, 8))
plt.plot(ct, pressure, linewidth=1, label='Temperature')

# Scatter masks: ml, int, cl
plt.scatter(ct[mask_ml], pressure[mask_ml], s=20, label='Mixed Layer (ml)')
plt.scatter(ct[mask_int], pressure[mask_int], s=20, label='Interface (int)')
# plt.scatter(ct[mask_cl], pressure[mask_cl], s=20, label='Connection Layer (cl)')

plt.scatter(ct[cl_mushy], pressure[cl_mushy], marker='*', s=60, label='CL – Mushy')
plt.scatter(ct[cl_supermushy], pressure[cl_supermushy], marker='x', s=60, label='CL – Supermushy')

plt.axhline(dmax, linestyle='--', label=f'Depth of Max CT: {dmax:.1f} m')
plt.axhline(dmin, linestyle='--', label=f'Depth of Min CT: {dmin:.1f} m')

# Aesthetics
plt.gca().invert_yaxis()
plt.xlabel('Conservative Temperature (°C)')
plt.ylabel('Pressure (m)')
plt.title(f'Temperature Profile {prof_no} of {itp} on {date_str} with masks')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
