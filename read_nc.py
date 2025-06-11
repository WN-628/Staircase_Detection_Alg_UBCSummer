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
ct = ds.variables["ct"][prof_no, :]               # Conservative temperature
sa = ds.variables["sa"][prof_no, :]               # Absolute salinity
pressure = ds.variables["pressure"][:]      # Pressure levels

mask_sf = ds.variables["mask_ml_sf"][prof_no, :]  # Salt-finger mask
mask_dc = ds.variables["mask_ml_dc"][prof_no, :]  # Diffusive convection mask

# Boolean masks
sf_mask = mask_sf > 0
dc_mask = mask_dc > 0

print(f"ct[{prof_no}, :] NaN count:", np.isnan(ct).sum())
print(f"sa[{prof_no}, :] NaN count:", np.isnan(sa).sum())
print(f"mask_ml_sf[{prof_no}, :] unique values:", np.unique(mask_sf))
print(f"mask_ml_dc[{prof_no}, :] unique values:", np.unique(mask_dc))


# Plot temperature and salinity
fig, axs = plt.subplots(1, 2, figsize=(12, prof_no), sharey=True)

# --- Temperature plot
axs[0].plot(ct, pressure, label="Temperature", color="black", linewidth=1)
axs[0].scatter(ct[sf_mask], pressure[sf_mask], color="blue", s=10, label="Salt-Finger")
axs[0].scatter(ct[dc_mask], pressure[dc_mask], color="red", s=10, label="Diffusive Convection")
axs[0].invert_yaxis()
axs[0].set_xlabel("Conservative Temperature (°C)")
axs[0].set_ylabel("Pressure (dbar)")
axs[0].set_title("Temperature Profile")
axs[0].legend()
axs[0].grid(True)

# --- Salinity plot
axs[1].plot(sa, pressure, label="Salinity", color="black", linewidth=1)
axs[1].scatter(sa[sf_mask], pressure[sf_mask], color="blue", s=10, label="Salt-Finger")
axs[1].scatter(sa[dc_mask], pressure[dc_mask], color="red", s=10, label="Diffusive Convection")
axs[1].set_xlabel("Absolute Salinity (g/kg)")
axs[1].set_title("Salinity Profile")
axs[1].legend()
axs[1].grid(True)

plt.suptitle("Staircase Structure (Profile {})".format(prof_no), fontsize=16)
fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()