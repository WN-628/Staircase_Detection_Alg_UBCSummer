import netCDF4 as netcdf
import numpy as np
import scipy.interpolate as interpolate
import scipy.stats as stats
import scipy.ndimage as ndimage
import os
import shutil
import zipfile

#import functions
from data_preparation import *
from create_netcdf import *
from staircase_detection_algorithm import *

from config import FIXED_RESOLUTION_DBAR

"""
Script to download profiles from ftp server, to interpolate them to a 1 dbar resolution and to detect 
thermohaline staircases in the interpolated profiles. A detailed description can be found in:

van der Boog, C.G. et al. (20xx), Global dataset of thermohaline staircases obtained from Argo
floats and Ice Tethered Profilers. Submitted to Earth System Science Data

made by: Carine van der Boog (2020)
"""
for flt_type in [1]:
  # copyright()
  print('Ice tethered profiles')
  list1 = [f for f in os.listdir() if f.endswith('.zip')]
  floats=np.array(list1) 

  for i in range(len(list1)):
    zip_name = os.path.splitext(list1[i])[0]
    ncfile = f"{zip_name}.nc"
    create_netcdf(ncfile, int(2000 / FIXED_RESOLUTION_DBAR))
    print(f"📦 Processing {list1[i]} → will save to {ncfile}")

  
    #download data from ftp server and interpolate to vertical resolution of 1 dbar
    index = np.where(floats==list1[i])[0][0]
    # docname1 = 'ftp://ftp.whoi.edu/whoinet/itpdata/'+list1[i]
    filename1 = list1[i]
    # wget.download(docname1)
        
    os.makedirs('tmp/')
    zip_ref = zipfile.ZipFile(floats[index],'r')
    zip_ref.extractall('tmp/')  
    os.remove(floats[index])
    
    profiles = []
    for root, _, files in os.walk('tmp/'):
        for f in files:
            if f.endswith('.csv'):
                profiles.append(os.path.join(root, f))
    prof_no, p, lat, lon, ct, sa, juld = load_data_csv_zip('', profiles, interp = True, resolution= FIXED_RESOLUTION_DBAR)
    # prof_no,p,lat,lon,ct,sa,juld = load_data_itp('tmp/',profiles,True)
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
    juld  = juld[~np.all(np.isnan(ct), axis=1)]
    prof_no = prof_no[~np.all(np.isnan(ct), axis=1)]
    ct    = ct[~np.all(np.isnan(ct), axis=1),:]

  if len(p)>0:
    if len(lat)==1 and p.max()>0:
      ct = np.ma.squeeze(ct)[np.newaxis,:]
      sa = np.ma.squeeze(sa)[np.newaxis,:]   
      p  = np.ma.squeeze(p)[np.newaxis,:]
    
    if p.max()>0:
      #detection algorithm
      c1 = 0.0005
      c2 = 0.005
      c3 = int(200 / FIXED_RESOLUTION_DBAR)  # Convert 200 dbar to index based on FIXED_RESOLUTION_DBAR
      c4 = int(30 / FIXED_RESOLUTION_DBAR)  # Convert 30 dbar to index based on FIXED_RESOLUTION_DBAR
      gl, ml, masks = get_mixed_layers(np.ma.copy(p),np.ma.copy(ct),np.ma.copy(sa),c1,c2,c3,c4) 

      fh2 = netcdf.Dataset(ncfile,'r+')
      t0 = len(fh2.variables['n'][:])
      t1 = len(fh2.variables['n'][:])+len(lat)
    
      #general
      fh2.variables['n'][t0:t1]                   = np.arange(len(lat),dtype=np.int32)
      fh2.variables['lat'][t0:t1]                 = lat
      fh2.variables['lon'][t0:t1]                 = lon
      fh2.variables['prof'][t0:t1]                = np.arange(len(lat))
      fh2.variables['juld'][t0:t1]                = juld
      fh2.variables['ct'][t0:t1,:]                = ct
      fh2.variables['sa'][t0:t1,:]                = sa
      fh2.variables['FloatID'][t0:t1]             = prof_no
      
      #masks
      fh2.variables['mask_gl_sf_layer'][t0:t1,:]  = masks.gl_sf_layer
      fh2.variables['mask_ml_sf_layer'][t0:t1,:]  = masks.ml_sf_layer
      fh2.variables['mask_gl_sf'][t0:t1,:]        = masks.gl_sf
      fh2.variables['mask_ml_sf'][t0:t1,:]        = masks.ml_sf
      fh2.variables['mask_gl_dc_layer'][t0:t1,:]  = masks.gl_dc_layer
      fh2.variables['mask_ml_dc_layer'][t0:t1,:]  = masks.ml_dc_layer
      fh2.variables['mask_gl_dc'][t0:t1,:]        = masks.gl_dc
      fh2.variables['mask_ml_dc'][t0:t1,:]        = masks.ml_dc

      # mixed layer characteristics
      fh2.variables['ml_h'][t0:t1,:]              = ml.height
      fh2.variables['ml_p'][t0:t1,:]              = ml.p
      fh2.variables['ml_T'][t0:t1,:]              = ml.T
      fh2.variables['ml_S'][t0:t1,:]              = ml.S
      fh2.variables['ml_Tu'][t0:t1,:]             = ml.Tu
      fh2.variables['ml_R'][t0:t1,:]              = ml.R
      fh2.variables['ml_r'][t0:t1,:]              = ml.r
      fh2.variables['ml_h'][t0:t1,:]              = ml.height
        
      #gradient layer characteristics
      fh2.variables['gl_dT'][t0:t1,:]             = gl.dT
      fh2.variables['gl_dS'][t0:t1,:]             = gl.dS
      fh2.variables['gl_dr'][t0:t1,:]             = gl.dr
      fh2.variables['gl_h'][t0:t1,:]              = gl.dist
      fh2.variables['gl_Tu'][t0:t1,:]             = gl.Tu
      fh2.variables['gl_R'][t0:t1,:]              = gl.R
      fh2.close()
        
