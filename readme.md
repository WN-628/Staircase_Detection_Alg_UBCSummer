# Thermohaline Staircase Detection Pipeline

## Overview

This repository implements an end-to-end workflow for detecting thermohaline staircases in oceanographic profile data:

1. **Load** raw profile files (from zipped .csv archives).
2. **Interpolate** each profile to a fixed vertical resolution (default 0.25 m).
3. **Detect** mixed layers, gradient (interface) layers, and full staircase structures.
4. **Store** raw data, masks, and depth extrema into NetCDF files.
5. **Visualize** individual profiles with mask overlays and annotations.

------

## Features

- **Fixed-resolution interpolation** of temperature & salinity profiles.
- **Flexible input formats**: zipped CSV (and easily extendable to MAT).
- **Mixed-layer & interface detection** via dual-threshold & run-length filtering.
- **Continuity & alternation filtering** to identify valid staircase structures.
- **NetCDF output** storing profile metadata, raw fields, masks, and extrema depths.
- **Quick-start plotting** utility for visual inspection of one profile at a time.

------

## Requirements

- Python ≥ 3.8
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [pandas](https://pandas.pydata.org/)
- [netCDF4](https://unidata.github.io/netcdf4-python/)
- [gsw (TEOS-10)](https://www.teos-10.org/)
- [matplotlib](https://matplotlib.org/)
- [h5py](https://www.h5py.org/) (optional, for MAT files)

Install all dependencies with:

```bash
pip install numpy scipy pandas netCDF4 gsw matplotlib h5py
```

------

## Configuration

All global settings live in **`config.py`**:

- `FIXED_RESOLUTION_METER`
   : Target vertical grid spacing (in meters) for interpolation.

Adjust this value to change interpolation resolution.

------

## File Structure

- **`config.py`**
   Defines global constants (e.g. `FIXED_RESOLUTION_METER`).
- **`create_netcdf.py`**
  - `create_netcdf(filename, max_count)`
  - Initializes a new NetCDF file with appropriate dimensions and variables.
- **`data_preparation.py`**
  - `load_data_csv_zip(path, profiles, interp, resolution)`
  - (Optional) `load_data_mat_zip(...)`
  - Reads raw CSV or MAT files, converts to CT/SA, interpolates to fixed grid.
- **`staircase_detector.py`**
  - `cent_derivative2d(f, z)`
  - `detect_mixed_layers_dual(...)`
  - `mask_continuity(mask, length)`
  - `continuity(arr, num_one, num_two, num_three)`
  - `get_mixed_layers(p, ct, thres_ml, thres_int, min_run, mushy)`
     Implements core detection logic and filters.
- **`main.py`**
   Orchestrates the full pipeline:
  1. Find `.zip` archives
  2. Create NetCDF via `create_netcdf`
  3. Unzip & load profiles (it would warns you if the .zip file contains file that is not .csv)
  4. Detect layers & staircases
  5. Write results to NetCDF
  6. Clean up temporary files
- **`read_nc.py`**
   Quick-plot script for one profile:
  1. Opens a NetCDF file
  2. Prints dimensions & variable shapes
  3. Extracts CT vs. depth, masks & extrema temperature
  4. Generates a matplotlib plot with overlays

------

## Usage

### 1. Prepare Input

Place your zipped CSV archives (each containing `*.csv` files) into the working directory.

### 2. Run the Pipeline

```bash
python main.py
```

For each `archive.zip`, this will generate `archive.nc` containing:

- Raw CT/SA profiles
- Boolean masks for mixed layers, interfaces, connection layers, and full staircases
- Depths of minimum & maximum CT points

### 3. Inspect Results

Use the quick-start script:

```bash
python read_nc.py --file archive.nc --profile 0
```

Or edit the `file_path` & `prof_no` variables at the top of the script. A plot of CT vs. depth with mask overlays will open.

------

## Configuration Parameters

Inside **`main.py`**, you can tweak detection thresholds:

- `thres_ml_upper`
   : Slope threshold for marking mixed-layer points.
- `thres_int_lower`
   : Slope threshold for marking interface (gradient) points.
- `layer_grid_length`
   : Minimum number of contiguous grid points for a valid layer
  - Note: I currently keep this for both mixed and interface, it can have separate standard easily
- `cl_length`
   : Maximum vertical gap (in meters) to link layers into a staircase.

Adjust these to tune sensitivity to your dataset.

------

## Extending & Testing

- **New Input Formats**
   Add `load_data_xyz(...)` in `data_preparation.py`.

- **Alternative Detection**
   Swap in your own detection function instead of `detect_mixed_layers_dual`.

- **Batch Visualization**
   Extend `read_nc.py` to loop over multiple profiles or archives.

- **Store needed information**

  - Add new boolean variable for each profile to show whether it has valid staircase structure or not
  - Build class structure for both mixed layer and interface

  