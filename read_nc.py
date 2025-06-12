import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

# Load the NetCDF file (adjust path as needed)
file_path = 'itp65cormat.nc'
ds = nc.Dataset(file_path)

# Print dimensions
print("=== Dimensions ===")
for dim_name, dim in ds.dimensions.items():
    print(f"{dim_name}: {len(dim)}")

# Print variables and their shapes
print("\n=== Variables ===")
for var_name, var in ds.variables.items():
    print(f"{var_name}: shape = {var.shape}, dtype = {var.dtype}")

prof_no = 18

# Extract data for profile 0
# ct = ds.variables["ct"][prof_no, :]               # Conservative temperature

ct_raw = ds.variables["ct"][prof_no, :]
pressure = ds.variables["pressure"][:]

# For plotting
ct = np.ma.masked_invalid(ct_raw)

# For dot plotting (full)
temp_max = np.nanmax(ct_raw)
temp_min = np.nanmin(ct_raw)

sa = ds.variables["sa"][prof_no, :]               # Absolute salinity
pressure = ds.variables["pressure"][:]      # Pressure levels
mask = ds.variables["mask_ml_sf_layer"][prof_no, :]  # Salt-finger mask

# New: Load max/min temperature depths
depth_max_T = ds.variables["depth_max_T"][prof_no]
depth_min_T = ds.variables["depth_min_T"][prof_no]
print(f"Depth of max temperature for profile {prof_no}: {depth_max_T} dbar")
print(f"Depth of min temperature for profile {prof_no}: {depth_min_T} dbar")
print(f"ct[{prof_no}, :] NaN count:", ct.mask.sum())
print(f"sa[{prof_no}, :] NaN count:", sa.mask.sum())
print(f"mask_ml[18, :] unique values:", np.unique(mask))

# Compute max/min T (handling full-masked cases)
if ct.count() > 0:
    temp_max = ct.max()
    temp_min = ct.min()
else:
    temp_max = np.nan
    temp_min = np.nan

# Boolean masks
valid = ~ct.mask
sf_mask = (mask > 0) & valid

print(f"ct[{prof_no}, :] NaN count:", np.isnan(ct).sum())
print(f"sa[{prof_no}, :] NaN count:", np.isnan(sa).sum())
print(f"mask_ml_sf[{prof_no}, :] unique values:", np.unique(mask))


# Plot temperature and salinity
fig, axs = plt.subplots(1, 2, figsize=(12, prof_no), sharey=True)

# --- Temperature plot
axs[0].plot(ct, pressure, label="Temperature", color="black", linewidth=1)
axs[0].scatter(ct[sf_mask], pressure[sf_mask], color="blue", s=10, label="Gradient Layer")

# Add min/max markers
# axs[0].scatter([temp_max], [depth_max_T], color="red", s=50, label="Max Temp", marker="o", zorder=5)
# axs[0].scatter([temp_min], [depth_min_T], color="green", s=50, label="Min Temp", marker="o", zorder=5)

axs[0].scatter(
    [temp_max], [depth_max_T],
    color="red", s=100, marker="o", edgecolors="black", linewidths=1.2,
    label=f"Max Temp: {temp_max:.2f} °C @ {depth_max_T:.1f} dbar", zorder=5
)

axs[0].scatter(
    [temp_min], [depth_min_T],
    color="green", s=100, marker="o", edgecolors="black", linewidths=1.2,
    label=f"Min Temp: {temp_min:.2f} °C @ {depth_min_T:.1f} dbar", zorder=5
)



axs[0].invert_yaxis()
axs[0].set_xlabel("Conservative Temperature (°C)")
axs[0].set_ylabel("Pressure (dbar)")
axs[0].set_title("Temperature Profile")
axs[0].legend()
axs[0].grid(True)

# --- Salinity plot
axs[1].plot(sa, pressure, label="Salinity", color="black", linewidth=1)
axs[1].scatter(sa[sf_mask], pressure[sf_mask], color="blue", s=10, label="Gradient Layer")
axs[1].set_xlabel("Absolute Salinity (g/kg)")
axs[1].set_title("Salinity Profile")
axs[1].legend()
axs[1].grid(True)

plt.suptitle("Staircase Structure (Profile {})".format(prof_no), fontsize=16)
# fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
plt.tight_layout(rect=[0, 0, 1, 0.96])

print("Plot x-limits:", axs[0].get_xlim())
print("Plot y-limits:", axs[0].get_ylim())
print("Max Temp:", temp_max, "@", depth_max_T)
print("Min Temp:", temp_min, "@", depth_min_T)

plt.show()