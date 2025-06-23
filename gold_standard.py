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

# Process each file
for fname in nc_files:
    basename = os.path.basename(fname)
    with nc.Dataset(fname, 'r') as ds:
        mask_sc = ds.variables['mask_sc'][:].astype(bool)
        cl_mushy = ds.variables['cl_mushy'][:].astype(bool)
        cl_supermushy = ds.variables['cl_supermushy'][:].astype(bool)
        float_ids = ds.variables['FloatID'][:]

        n_profiles = mask_sc.shape[0]
        total_profiles += n_profiles

        # Determine profiles with any staircase
        has_sc = np.any(mask_sc, axis=1)
        sc_indices = np.where(has_sc)[0]
        count_sc = len(sc_indices)
        profiles_with_staircase += count_sc

        # Determine profiles with cl_mushy
        has_cl_m = np.any(cl_mushy, axis=1)
        count_cl_m = np.count_nonzero(has_cl_m)
        profiles_with_cl_mushy += count_cl_m

        # Determine profiles with cl_supermushy
        has_cl_sm = np.any(cl_supermushy, axis=1)
        count_cl_sm = np.count_nonzero(has_cl_sm)
        profiles_with_cl_supermushy += count_cl_sm

        # Profiles with staircase but no connection layer
        no_cl_mask = has_sc & ~has_cl_m & ~has_cl_sm
        no_cl_indices = np.where(no_cl_mask)[0]
        count_no_cl = len(no_cl_indices)
        profiles_stair_no_cl += count_no_cl

        # Capture the FloatID values for these profiles
        stairs_no_cl_ids = float_ids[no_cl_indices] if count_no_cl > 0 else []
        if count_no_cl > 0:
            stairs_no_cl_list[basename] = stairs_no_cl_ids.tolist()

        # # Print per-file summary
        # print(
        #     f"{basename}: {n_profiles} profiles, "
        #     f"{count_sc} with staircases, "
        #     f"{count_no_cl} with staircases but no cl, "
        #     f"{count_cl_m} with cl_mushy, "
        #     f"{count_cl_sm} with cl_supermushy"
        # )

# Print detailed list of ITPs and profile IDs without connection layers
if stairs_no_cl_list:
    print("\nProfiles with staircases but no connection layers:")
    for fname, id_list in stairs_no_cl_list.items():
        print(f"- {fname}: {len(id_list)} profiles -> IDs: {id_list}")
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
