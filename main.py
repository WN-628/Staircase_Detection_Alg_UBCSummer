import netCDF4 as netcdf
import numpy as np
import scipy.interpolate as interpolate
import scipy.stats as stats
import scipy.ndimage as ndimage
import os
import shutil
import zipfile
import warnings

#import functions
from data_preparation import load_data_csv_zip
from create_netcdf import *
# from staircase_detector import get_mixed_layers
from staircase_detector import get_mixed_layers
from config import FIXED_RESOLUTION_METER

"""
Script to detect 
thermohaline staircases in the interpolated profiles with a fixed resolution in depth (0.25m). 

This code is based on the ideas presented in:
van der Boog, C.G. et al. (20xx), intobal dataset of thermohaline staircases obtained from Argo
floats and Ice Tethered Profilers. 

and 

Kat's Staircase Detection Algorithm. 

made by: Yujun Ling at UBC (Summer 2025)
"""

print('Ice tethered profiles')
list1 = [f for f in os.listdir() if f.endswith('.zip')]
floats=np.array(list1) 

# Parameters for the staircase detection algorithm
depth_thres = 450       # minimum depth threshold for valid profiles (in m)
thres_ml_upper = 0.001  # gradient threshold for mixed layer detection 
thres_int_lower = 0.005 # gradient threshold for interface detection
layer_grid_length = 5   # minumum length of mixed layer in grid points (0.25m resolution)
cl_length = 1.0         # maximum allowed length in meters 
smooth_length = 7       # smoothing length in grid points (0.25m resolution)

for i in range(len(list1)):
  zip_name = os.path.splitext(list1[i])[0]
  ncfile = f"{zip_name}.nc"
  create_netcdf(ncfile, int(2000 / FIXED_RESOLUTION_METER))
  print(f"📦 Processing {list1[i]} → will save to {ncfile}")

  index = np.where(floats==list1[i])[0][0]
  filename1 = list1[i]
  
  os.makedirs('tmp/')
  zip_ref = zipfile.ZipFile(floats[index],'r')
  zip_ref.extractall('tmp/')  
  os.remove(floats[index])
  
  profiles = []
  for root, _, files in os.walk('tmp/'):
      for f in files:
          if f.endswith('.csv'):
              profiles.append(os.path.join(root, f))
          else: 
              warnings.warn(f"File {f} is not a CSV file and will be ignored.")
  
  
  prof_no, p, lat, lon, ct, sa, dates = load_data_csv_zip('', profiles, interp = False, resolution= FIXED_RESOLUTION_METER, depth_thres=depth_thres)
  
  shutil.rmtree('tmp/')  
  # print("Loaded profile count:", len(lat))
  # print("Latitude:", lat)
  # print("CT shape:", ct.shape)
  # print("Max pressure:", p.max() if isinstance(p, np.ndarray) else "Invalid")
  # print("Any non-NaN CT rows:", np.any(~np.all(np.isnan(ct), axis=1)))

  if len(lat)>1:
    #remove profiles with only nans
    n     = np.arange(len(lat),dtype=np.int32)
    n     = n[~np.all(np.isnan(ct), axis=1)]
    p     = p[~np.all(np.isnan(ct), axis=1),:]
    lon   = lon[~np.all(np.isnan(ct), axis=1)]
    lat   = lat[~np.all(np.isnan(ct), axis=1)]
    sa    = sa[~np.all(np.isnan(ct), axis=1),:]
    dates  = dates[~np.all(np.isnan(ct), axis=1)]
    prof_no = prof_no[~np.all(np.isnan(ct), axis=1)]
    ct    = ct[~np.all(np.isnan(ct), axis=1),:]

  if len(p)>0:
    if len(lat)==1 and p.max()>0:
      ct = np.ma.squeeze(ct)[np.newaxis,:]
      sa = np.ma.squeeze(sa)[np.newaxis,:]   
      p  = np.ma.squeeze(p)[np.newaxis,:]
    
    if p.max()>0:
      #detection algorithm
      masks, depth_min_T, depth_max_T = get_mixed_layers(np.ma.copy(p),np.ma.copy(ct),thres_ml_upper,thres_int_lower, layer_grid_length, cl_length, smooth_length) 

      fh2 = netcdf.Dataset(ncfile,'r+')
      t0 = len(fh2.variables['n'][:])
      t1 = len(fh2.variables['n'][:])+len(lat)
    
      #general
      fh2.variables['n'][t0:t1]                   = np.arange(len(lat),dtype=np.int32)
      fh2.variables['lat'][t0:t1]                 = lat
      fh2.variables['lon'][t0:t1]                 = lon
      fh2.variables['prof'][t0:t1]                = np.arange(len(lat))
      fh2.variables['dates'][t0:t1]               = dates
      fh2.variables['ct'][t0:t1,:]                = ct
      fh2.variables['sa'][t0:t1,:]                = sa
      fh2.variables['FloatID'][t0:t1]             = prof_no
      
      #masks
      fh2.variables['mask_int'][t0:t1,:]  = masks.int
      fh2.variables['mask_ml'][t0:t1,:]  = masks.ml
      fh2.variables['mask_cl'][t0:t1,:]  = masks.cl
      fh2.variables['mask_sc'][t0:t1,:]  = masks.sc
      assert np.any(masks.ml), "No mixed layer mask found in the data."
      assert np.any(masks.int), "No interface mask found in the data."
      
      # temperature max and min over depths
      fh2.variables['depth_max_T'][t0:t1] = depth_max_T
      fh2.variables['depth_min_T'][t0:t1] = depth_min_T
      
      fh2.close()

      # # mixed layer characteristics
      # fh2.variables['ml_h'][t0:t1,:]              = ml.height
      # fh2.variables['ml_p'][t0:t1,:]              = ml.p
      # fh2.variables['ml_T'][t0:t1,:]              = ml.T
      # fh2.variables['ml_S'][t0:t1,:]              = ml.S
      # fh2.variables['ml_Tu'][t0:t1,:]             = ml.Tu
      # fh2.variables['ml_R'][t0:t1,:]              = ml.R
      # fh2.variables['ml_r'][t0:t1,:]              = ml.r
      # fh2.variables['ml_h'][t0:t1,:]              = ml.height
      
      # #interface characteristics
      # fh2.variables['int_dT'][t0:t1,:]             = int.dT
      # fh2.variables['int_dS'][t0:t1,:]             = int.dS
      # fh2.variables['int_dr'][t0:t1,:]             = int.dr
      # fh2.variables['int_h'][t0:t1,:]              = int.dist
      # fh2.variables['int_Tu'][t0:t1,:]             = int.Tu
      # fh2.variables['int_R'][t0:t1,:]              = int.R
      
      
