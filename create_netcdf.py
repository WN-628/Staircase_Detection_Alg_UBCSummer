import netCDF4 as netcdf
import numpy as np
import time

from config import FIXED_RESOLUTION_METER

def create_netcdf(filename,max_count):
  
  # --- Dimensions ---
  
  fh2 = netcdf.Dataset(filename,'w',format='NETCDF4')
  fh2.createDimension('Nobs',None)
  x1 = fh2.createVariable('n', np.int32, ('Nobs'))
  x1.long_name     = 'Profile'
  x1.standard_name = 'no'
  
  pressure_levels = np.arange(0, 2000, FIXED_RESOLUTION_METER)
  
  fh2.createDimension('pressure', len(pressure_levels))
  x1 = fh2.createVariable('pressure', 'f4', ('pressure'))
  x1.long_name     = 'Pressure'
  x1.standard_name = 'depth'
  x1.units         = 'meter'
  x1[:]            = pressure_levels

  fh2.createDimension('mixed_layers',max_count)
  x1 = fh2.createVariable('mixed_layers', 'f4', ('mixed_layers'),fill_value=0)
  x1.long_name     = 'Mixed-Layer-count'
  x1.standard_name = 'MLD'
  x1[:]            = np.arange(0,int(max_count))

  fh2.createDimension('gradient_layers',max_count+1)
  x1 = fh2.createVariable('gradient_layers', 'f4', ('gradient_layers'),fill_value=0)
  x1.long_name     = 'Gradient-Layer-count'
  x1.standard_name = 'MLD'
  x1[:]            = np.arange(0,int(max_count)+1)
  
  fh2.createDimension('staircase_structures',max_count)
  x1 = fh2.createVariable('staircase_structures', 'f4', ('staircase_structures'),fill_value=0)
  x1.long_name     = 'Staircase-Structure-count'
  x1.standard_name = 'staircase_structures'
  x1[:]            = np.arange(0,int(max_count))
  
  # --- Variables ---
  
  x2 = fh2.createVariable('prof',np.int32, ('Nobs'),fill_value=0,zlib=True)
  x2.long_name     = 'Profile number of float'
  x2.standard_name = 'prof'
  
  x2 = fh2.createVariable('FloatID',np.int32, ('Nobs'),fill_value=0,zlib=True)
  x2.long_name     = 'Float ID'
  x2.standard_name = 'FloatID'
  
  x2 = fh2.createVariable('dates','f8',('Nobs',),fill_value=0,zlib=True)
  x2.long_name     = 'Profile date'
  x2.standard_name = 'dates'
  x2.units         = 'seconds since 1970-01-01T00:00:00Z'
  x2.calendar      = 'gregorian'
  
  x2 = fh2.createVariable('lon','f4',('Nobs',),fill_value=0,zlib=True)
  x2.long_name     = 'Longitude of float'
  x2.standard_name = 'lon'
  x2.units         = 'degrees'
  
  x2 = fh2.createVariable('lat','f4',('Nobs',),fill_value=0,zlib=True)
  x2.long_name     = 'Latitude of float'
  x2.standard_name = 'lat'
  x2.units         = 'degrees'
  
  # min and max temperature
  x2 = fh2.createVariable('depth_max_T', np.float32, ('Nobs',), fill_value=np.nan)
  x2.long_name = 'Depth of maximum temperature'
  x2.standard_name = 'max_depth_T'
  x2.units         = 'degrees Celsius'

  x2 = fh2.createVariable('depth_min_T', np.float32, ('Nobs',), fill_value=np.nan)
  x2.long_name = 'Depth of minimum temperature'
  x2.standard_name = 'min_depth_T'
  x2.units         = 'degrees Celsius'

  # other variables
  x2 = fh2.createVariable('ct','f8',('Nobs','pressure'),fill_value=np.nan,zlib=True)
  x2.long_name     = 'Conservative Temperature'
  x2.standard_name = 'ct'
  x2.units         = 'degrees Celsius'

  x2 = fh2.createVariable('sa','f8',('Nobs','pressure'),fill_value=np.nan,zlib=True)
  x2.long_name     = 'Absolute Salinity'
  x2.standard_name = 'sa'
  x2.units         = 'g/kg'

  # masks variables
  x2 = fh2.createVariable('mask_int',np.int32,('Nobs','pressure'),fill_value=0,zlib=True)
  x2.long_name     = 'mask with sequences of interfaces'
  x2.standard_name = 'mask_int'
  x2.units         = ' '

  x2 = fh2.createVariable('mask_ml',np.int32,('Nobs','pressure'),fill_value=0,zlib=True)
  x2.long_name     = 'mask with sequences of mixed layers'
  x2.standard_name = 'mask_ml'
  x2.units         = ' '
  
  x2 = fh2.createVariable('mask_cl',np.int32,('Nobs','pressure'),fill_value=0,zlib=True)
  x2.long_name     = 'mask with sequences of connecting layers'
  x2.standard_name = 'mask_cl'
  x2.units         = ' '
  
  x2 = fh2.createVariable('mask_sc',np.int32,('Nobs','pressure'),fill_value=0,zlib=True)
  x2.long_name     = 'mask with sequences of staircase structure'
  x2.standard_name = 'mask_sc'
  x2.units         = ' '
  
  
  
  
  
  fh2.close()