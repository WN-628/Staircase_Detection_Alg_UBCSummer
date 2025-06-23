import glob
import netCDF4 as nc
import numpy as np
import os

# Directory containing production .nc files
PROD_DIR = 'prod_files'

# Gather all .nc files from the production directory
nc_files = glob.glob(os.path.join(PROD_DIR, '*.nc'))
if not nc_files:
    print(f"❌ No .nc files found in '{PROD_DIR}'")
    exit(1)

# Counters for overall summary
total_profiles = 0
profiles_with_staircase = 0
profiles_stair_no_cl = 0
profiles_with_cl_mushy = 0
profiles_with_cl_supermushy = 0

# Store lists of profiles without any connection layers
stairs_no_cl_list = {}

for fname in nc_files:
    basename = os.path.basename(fname)
    with nc.Dataset(fname, 'r') as ds:
        # vlen masks: each ds.variables['mask_sc'][i] is a 1D int8 array
        mask_sc_var       = ds.variables['mask_sc']
        cl_mushy_var      = ds.variables['cl_mushy']
        cl_supermushy_var = ds.variables['cl_supermushy']
        float_ids         = ds.variables['FloatID'][:]

        n_profiles = mask_sc_var.shape[0]
        total_profiles += n_profiles

        # Determine for each profile whether it has any staircase points
        has_sc = []
        for i in range(n_profiles):
            arr = np.array(mask_sc_var[i], dtype=bool)   # vlen → nd array
            has_sc.append(arr.any())
        count_sc = sum(has_sc)
        profiles_with_staircase += count_sc

        # Determine for each profile whether it has any mushy connection layers
        has_cl_m = []
        for i in range(n_profiles):
            arr = np.array(cl_mushy_var[i], dtype=bool)
            has_cl_m.append(arr.any())
        count_cl_m = sum(has_cl_m)
        profiles_with_cl_mushy += count_cl_m

        # Determine for each profile whether it has any super-mushy connection layers
        has_cl_sm = []
        for i in range(n_profiles):
            arr = np.array(cl_supermushy_var[i], dtype=bool)
            has_cl_sm.append(arr.any())
        count_cl_sm = sum(has_cl_sm)
        profiles_with_cl_supermushy += count_cl_sm

        # Profiles with staircases but *no* connection layers at all
        no_cl_mask = [has_sc[i] and not has_cl_m[i] and not has_cl_sm[i]
                      for i in range(n_profiles)]
        no_cl_indices = [i for i, flag in enumerate(no_cl_mask) if flag]
        count_no_cl = len(no_cl_indices)
        profiles_stair_no_cl += count_no_cl

        # Record their FloatID values
        if count_no_cl:
            stairs_no_cl_list[basename] = float_ids[no_cl_indices].tolist()

# Print detailed list
if stairs_no_cl_list:
    print("\nProfiles with staircases but no connection layers:")
    for fname, id_list in stairs_no_cl_list.items():
        print(f"- {fname}: {len(id_list)} profiles → IDs: {id_list}")
else:
    print("\nNo profiles found with staircases but no connection layers.")

# Summary
print("\n—— Summary ——")
print(f"Total profiles scanned               : {total_profiles}")
print(f"Profiles with ≥1 staircase mask      : {profiles_with_staircase}")
print(f"Profiles with staircases but no cl   : {profiles_stair_no_cl}")
print(f"Profiles with cl_mushy               : {profiles_with_cl_mushy}")
print(f"Profiles with cl_supermushy          : {profiles_with_cl_supermushy}")
print(f"Profiles without staircases          : {total_profiles - profiles_with_staircase}")
