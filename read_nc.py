import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

# Configuration
file_path = 'itp65cormat.nc'  # Update with your NetCDF file path
ds = nc.Dataset(file_path)

prof_no = 853                  # Profile index to plot

profile_ids = ds.variables['FloatID'][:]  # 1D array of profile IDs
if prof_no in profile_ids:
    prof_idx = int(np.where(profile_ids == prof_no)[0][0])
else:
    raise ValueError(f"Profile ID {prof_no} not found; available IDs: {profile_ids}")

# Print dimensions
print("=== Dimensions ===")
for dim_name, dim in ds.dimensions.items():
    print(f"{dim_name}: {len(dim)}")

# Print variables and their shapes
print("\n=== Variables ===")
for var_name, var in ds.variables.items():
    print(f"{var_name}: shape = {var.shape}, dtype = {var.dtype}")

# Load dataset
pressure = ds.variables['pressure'][:]
ct_raw = ds.variables['ct'][prof_idx, :]
mask_ml = ds.variables['mask_ml'][prof_idx, :].astype(bool)
mask_int = ds.variables['mask_int'][prof_idx, :].astype(bool)
mask_cl = ds.variables['mask_cl'][prof_idx, :].astype(bool)

# Mask invalid temperature
ct = np.ma.masked_invalid(ct_raw)

# Basic assertions 
assert np.any(mask_ml), "No mixed layer mask found in the data."
assert np.any(mask_int), "No interface mask found in the data."
# assert np.any(mask_cl), "No connection layer mask found in the data."

# Load extrema depths
depth_max_T = ds.variables['depth_max_T'][prof_idx]
depth_min_T = ds.variables['depth_min_T'][prof_idx]

# print("t_max_depth:", depth_max_T, "  closest grid points:", 
#         pressure[np.argsort(np.abs(pressure - depth_max_T))[:3]])

# print("t_max_depth:", depth_min_T, "  closest grid points:", 
#         pressure[np.argsort(np.abs(pressure - depth_min_T))[:3]])

# assert not np.isnan(depth_max_T), f"No maximum-temperature depth for profile ID {prof_no}"
# assert not np.isnan(depth_min_T), f"No minimum-temperature depth for profile ID {prof_no}"

# Find temperature values at extrema (due to 0.25m resolution, we can only use the closest grid points)
i_max = np.argmin(np.abs(pressure - depth_max_T))
i_min = np.argmin(np.abs(pressure - depth_min_T))
idx_max = np.array([i_max])
idx_min = np.array([i_min])
temp_max = ct[idx_max][0]
temp_min = ct[idx_min][0]

# Plot temperature profile
plt.figure(figsize=(6, 8))
plt.plot(ct, pressure, linewidth=1, label='Temperature')

# Scatter masks: ml, int, cl
plt.scatter(ct[mask_ml], pressure[mask_ml], s=20, label='Mixed Layer (ml)')
plt.scatter(ct[mask_int], pressure[mask_int], s=20, label='Interface (int)')
plt.scatter(ct[mask_cl], pressure[mask_cl], s=20, label='Connection Layer (cl)')

# Plot extrema points
if temp_max is not None:
    plt.scatter([temp_max], [depth_max_T], marker='o', s=50, label='Max Temp')
    plt.annotate('Max Temp', xy=(temp_max, depth_max_T), 
                xytext=(5, -10), textcoords='offset points')
if temp_min is not None:
    plt.scatter([temp_min], [depth_min_T], marker='o', s=50, label='Min Temp')
    plt.annotate('Min Temp', xy=(temp_min, depth_min_T), 
                xytext=(5, 10), textcoords='offset points')

# Aesthetics
plt.gca().invert_yaxis()
plt.xlabel('Conservative Temperature (°C)')
plt.ylabel('Pressure (dbar)')
plt.title(f'Temperature Profile with ml/int/cl Masks (Profile {prof_no})')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
