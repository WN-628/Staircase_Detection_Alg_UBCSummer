import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

# Configuration
file_path = 'itp65cormat.nc'  # Update with your NetCDF file path
prof_no = 18                  # Profile index to plot

# Load dataset
ds = nc.Dataset(file_path)
pressure = ds.variables['pressure'][:]
ct = ds.variables['ct'][prof_no, :]
mask_ml = ds.variables['mask_ml'][prof_no, :].astype(bool)
mask_gl = ds.variables['mask_gl'][prof_no, :].astype(bool)
mask_cl = ds.variables['mask_cl'][prof_no, :].astype(bool)

# Load extrema depths
t_max_depth = ds.variables['depth_max_T'][prof_no]
t_min_depth = ds.variables['depth_min_T'][prof_no]
# Find temperature values at extrema
idx_max = np.where(pressure == t_max_depth)[0]
idx_min = np.where(pressure == t_min_depth)[0]
temp_max = ct[idx_max][0] if idx_max.size else None
temp_min = ct[idx_min][0] if idx_min.size else None

# Mask invalid temperature
ct = np.ma.masked_invalid(ct)

# Plot temperature profile
plt.figure(figsize=(6, 8))
plt.plot(ct, pressure, linewidth=1, label='Temperature')

# Scatter masks: ml, gl, cl
plt.scatter(ct[mask_ml], pressure[mask_ml], s=20, label='Mixed Layer (ml)')
plt.scatter(ct[mask_gl], pressure[mask_gl], s=20, label='Gradient Layer (gl)')
plt.scatter(ct[mask_cl], pressure[mask_cl], s=20, label='Connection Layer (cl)')

# Plot extrema points
if temp_max is not None:
    plt.scatter([temp_max], [t_max_depth], marker='o', s=50, label='Max Temp')
    plt.annotate('Max Temp', xy=(temp_max, t_max_depth), 
                xytext=(5, -10), textcoords='offset points')
if temp_min is not None:
    plt.scatter([temp_min], [t_min_depth], marker='o', s=50, label='Min Temp')
    plt.annotate('Min Temp', xy=(temp_min, t_min_depth), 
                xytext=(5, 10), textcoords='offset points')

# Aesthetics
plt.gca().invert_yaxis()
plt.xlabel('Conservative Temperature (°C)')
plt.ylabel('Pressure (dbar)')
plt.title(f'Temperature Profile with ml/gl/cl Masks (Profile {prof_no})')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
