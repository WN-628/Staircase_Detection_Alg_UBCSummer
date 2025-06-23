import glob
import netCDF4 as nc
import numpy as np
import os

total_profiles = 0
profiles_with_staircase = 0

# This script scans all .nc files in the current directory to count profiles
# with at least one staircase mask.

# Loop over every .nc file in the current folder
for fname in glob.glob('*.nc'):
    with nc.Dataset(fname, 'r') as ds:
        mask_sc = ds.variables['mask_sc'][:].astype(bool)

        n_profiles = mask_sc.shape[0]
        total_profiles += n_profiles

        has_sc = np.count_nonzero(np.any(mask_sc, axis=1))
        profiles_with_staircase += has_sc

        print(f"{os.path.basename(fname)}: {n_profiles} profiles, {has_sc} with staircases")

print("\n—— Summary ——")
print(f"Total profiles scanned         : {total_profiles}")
print(f"Profiles with ≥1 staircase mask: {profiles_with_staircase}")
print(f"Profiles without staircases    : {total_profiles - profiles_with_staircase}")
2