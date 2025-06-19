import netCDF4 as nc
import numpy as np
from after_detector import count, extract_length, extract_temp_width
import matplotlib.pyplot as plt

# 1. Open the file you wrote in main.py
ds = nc.Dataset('itp65cormat.nc', 'r')
# print(list(ds.variables.keys()))

# 3. Read it properly
time_var = ds.variables['dates']       # a netCDF4.Variable
raw      = time_var[:]                 # extracts the array

# 4. If it has a CF‐time units attribute, convert to datetimes
if hasattr(time_var, 'units'):
    dates = nc.num2date(raw, time_var.units)
else:
    dates = raw                       # already in a directly comparable form
    
def to_ymd(dt):
    try:
        # most cftime/datetime objects support strftime
        return dt.strftime('%Y-%m-%d')
    except AttributeError:
        # fallback for cftime if strftime isn’t available
        return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}"

# compute earliest and latest
earliest = min(dates)
latest   = max(dates)

print("Earliest profile date:", to_ymd(earliest))
print("Latest   profile date:", to_ymd(latest))

# read in lon/lat
lon = ds.variables['lon'][:]
lat = ds.variables['lat'][:]

# scatter of profile locations
plt.figure(figsize=(6,6))
plt.scatter(lon, lat, s=20, alpha=0.7)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Profile Locations (lat vs. lon) for ITP65Cormat')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 2. Read your stored masks and data
mask_ml  = ds.variables['mask_ml'][:] .astype(bool)
mask_int = ds.variables['mask_int'][:].astype(bool)
mask_cl  = ds.variables['mask_cl'][:] .astype(bool)
mask_sc  = ds.variables['mask_sc'][:] .astype(bool)

p  = ds.variables['pressure'][:]   # depth or pressure [m]
ct = ds.variables['ct'][:]         # conservative temperature [°C]

# 3. Now run your layer‐analysis functions on the stored masks:

# Mixed‐layers
n_ml, ml_start, ml_end   = count(mask_ml)
ml_thickness             = extract_length(mask_ml, p)
ml_temp_width            = extract_temp_width(mask_ml, ct)

# Interfaces
n_int, int_start, int_end = count(mask_int)
int_thickness             = extract_length(mask_int, p)
int_temp_width            = extract_temp_width(mask_int, ct)

# Staircase-structures 
n_sc, sc_start, sc_end = count(mask_sc)

num_ml = sum(n_ml)
num_int = sum(n_int)
num_sc = sum(n_sc)

print("In ITP65CORMAT:")
total_profiles = mask_ml.shape[0]
print("Total profiles:", total_profiles)

profiles_with_sc = sum(1 for ns in n_sc if ns > 0)
print("Profiles with staircase detected:", profiles_with_sc)

print("Mixed Layer Counts:", num_ml)
print("Interface Counts:", num_int)
print("Staircase Counts:", num_sc)

# print("Mixed Layer Thicknesses:", ml_thickness)

def plot_histogram(data, bin_width, xlabel, title):
    """
    Plot a histogram for a list-of-lists of numeric values.
    
    Parameters
    ----------
    data : list of lists of float
        Each sublist contains values for one profile.
    bin_width : float
        Width of each histogram bin.
    xlabel : str
        Label for the x-axis.
    title : str
        Plot title.
    """
    # Flatten the list of lists
    flat = [t for prof in data for t in prof]
    if not flat:
        print(f"No data to plot for '{title}'.")
        return
    
    # Define bins from min to max
    bins = np.arange(min(flat), max(flat) + bin_width, bin_width)
    
    # Plot
    plt.figure()
    plt.hist(flat, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# # Mixed-layer thickness, 2 m bins
# plot_histogram(
#     ml_thickness, 
#     bin_width=2,
#     xlabel='Mixed-Layer Thickness (m)', 
#     title='Histogram of Mixed-Layer Thickness (2 m bins)'
# )

# # Interface temperature width, 0.05 °C bins
# plot_histogram(
#     int_temp_width, 
#     bin_width=0.005,
#     xlabel='Interface Layer Temperature Width (°C)', 
#     title='Histogram of Interface Layer Temperature Width (0.05 °C bins)'
# )

# 1) Flatten your list-of-lists
ml_flat     = [t for prof in ml_thickness    for t in prof]
int_tw_flat = [t for prof in int_temp_width  for t in prof]

# 2) Define bins for each
bin_ml_width = 2          # 2 m bins
bins_ml = np.arange(0, max(ml_flat,     default=0) + bin_ml_width, bin_ml_width) 
bin_tw_width = 0.005      # 0.05 °C bins
bins_tw = np.arange(0, max(int_tw_flat, default=0) + bin_tw_width, bin_tw_width) 

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# --- Mixed‐layer thickness ---
n_ml, bins_ml, patches_ml = ax1.hist(ml_flat, bins=bins_ml)
ax1.set_title('Mixed‐Layer Thickness')
ax1.set_xlabel('Thickness (m)')
ax1.set_ylabel('Frequency')

# annotate
for rect, count in zip(patches_ml, n_ml):
    height = rect.get_height()
    if height > 0:
        ax1.text(
            rect.get_x() + rect.get_width()/2,  # x = center of bar
            height,                             # y = top of bar
            str(int(count)),                    # label = count
            ha='center', va='bottom',
            fontsize=8
        )

# --- Interface temperature width ---
n_tw, bins_tw, patches_tw = ax2.hist(int_tw_flat, bins=bins_tw)
ax2.set_title('Interface Temperature Width')
ax2.set_xlabel('Width (°C)')

# annotate
for rect, count in zip(patches_tw, n_tw):
    height = rect.get_height()
    if height > 0:
        ax2.text(
            rect.get_x() + rect.get_width()/2,
            height,
            str(int(count)),
            ha='center', va='bottom',
            fontsize=8
        )

# Add a big title
fig.suptitle('Histogram for ITP65Cormat', fontsize=18)

fig.subplots_adjust(top=0.90)

# Tweak layout so the suptitle and subplots don’t collide
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()

ds.close()
