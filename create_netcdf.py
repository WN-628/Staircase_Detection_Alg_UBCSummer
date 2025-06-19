import netCDF4 as netcdf
import numpy as np
import time

from config import FIXED_RESOLUTION_METER

def create_netcdf(filename,max_count):
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

  fh2.createDimension('gradient_layers',max_count-1)
  x1 = fh2.createVariable('gradient_layers', 'f4', ('gradient_layers'),fill_value=0)
  x1.long_name     = 'Gradient-Layer-count'
  x1.standard_name = 'MLD'
  x1[:]            = np.arange(0,int(max_count)-1)
  
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
  x2.calendar     = 'gregorian'
  
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

  # # mixed layers
  # x2 = fh2.createVariable('ml_T','f4',('Nobs','mixed_layers'),fill_value=0,zlib=True)
  # x2.long_name     = 'average temperature of the mixed layer'
  # x2.standard_name = 'ml_T'
  # x2.units         = 'degrees Celsius'
 
  # x2 = fh2.createVariable('ml_S','f4',('Nobs','mixed_layers'),fill_value=0,zlib=True)
  # x2.long_name     = 'average salinity of the mixed layer'
  # x2.standard_name = 'ml_S'
  # x2.units         = 'g kg-1'
 
  # x2 = fh2.createVariable('ml_r','f4',('Nobs','mixed_layers'),fill_value=0,zlib=True)
  # x2.long_name     = 'average density of the mixed layer (sigma1)'
  # x2.standard_name = 'ml_r'
  # x2.units         = 'kg m-3'

  # x2 = fh2.createVariable('ml_p','f4',('Nobs','mixed_layers'),fill_value=0,zlib=True)
  # x2.long_name     = 'average pressure of the mixed layer'
  # x2.standard_name = 'ml_p'
  # x2.units         = 'dbar'

  # x2 = fh2.createVariable('ml_h','f4',('Nobs','mixed_layers'),fill_value=0,zlib=True)
  # x2.long_name     = 'height of the mixed layer'
  # x2.standard_name = 'ml_h'
  # x2.units         = 'dbar'
 
  # x2 = fh2.createVariable('ml_Tu','f4',('Nobs','mixed_layers'),fill_value=0,zlib=True)
  # x2.long_name     = 'average Turner aninte of the mixed layer'
  # x2.standard_name = 'ml_Tu'
  # x2.units         = 'degrees'

  # x2 = fh2.createVariable('ml_R','f4',('Nobs','mixed_layers'),fill_value=0,zlib=True)
  # x2.long_name     = 'average density ratio of the mixed layer'
  # x2.standard_name = 'ml_R'
  # x2.units         = ' '

  # #interface variables
  # x2 = fh2.createVariable('int_dT','f4',('Nobs','gradient_layers'),fill_value=0,zlib=True)
  # x2.long_name     = 'temperature difference in interface'
  # x2.standard_name = 'int_dT'
  # x2.units         = 'degrees Celsius'

  # x2 = fh2.createVariable('int_dS','f4',('Nobs','gradient_layers'),fill_value=0,zlib=True)
  # x2.long_name     = 'salinity difference in interface'
  # x2.standard_name = 'int_dS'
  # x2.units         = 'g kg-1'

  # x2 = fh2.createVariable('int_dr','f4',('Nobs','gradient_layers'),fill_value=0,zlib=True)
  # x2.long_name     = 'density difference in interface (sigma1)'
  # x2.standard_name = 'int_dr'
  # x2.units         = 'kg m-3 dbar-1'

  # x2 = fh2.createVariable('int_h','f4',('Nobs','gradient_layers'),fill_value=0,zlib=True)
  # x2.long_name     = 'interface height'
  # x2.standard_name = 'int_h'
  # x2.units         = 'dbar'

  # x2 = fh2.createVariable('int_Tu','f4',('Nobs','gradient_layers'),fill_value=0,zlib=True)
  # x2.long_name     = 'average Turner aninte of the interface'
  # x2.standard_name = 'int_Tu'
  # x2.units         = 'degrees'

  # x2 = fh2.createVariable('int_R','f4',('Nobs','gradient_layers'),fill_value=0,zlib=True)
  # x2.long_name     = 'average density ratio of the interface'
  # x2.standard_name = 'int_R'
  # x2.units         = ' '

  fh2.close()





  # x2 = fh2.createVariable('mask_int_sf_layer',np.int32,('Nobs','gradient_layers'),fill_value=0,zlib=True)
  # x2.long_name     = 'mask with sequences of interfaces in salt-finger regime'
  # x2.standard_name = 'mask_int_sf_layer'
  # x2.units         = ' '

  # x2 = fh2.createVariable('mask_ml_sf_layer',np.int32,('Nobs','mixed_layers'),fill_value=0,zlib=True)
  # x2.long_name     = 'mask with sequences of mixed layers in salt-finger regime'
  # x2.standard_name = 'mask_ml_sf_layer'
  # x2.units         = ' '

  # x2 = fh2.createVariable('mask_int_sf',np.int32,('Nobs','pressure'),fill_value=0,zlib=True)
  # x2.long_name     = 'big mask with sequences of interfaces in salt-finger regime'
  # x2.standard_name = 'mask_int_sf'
  # x2.units         = ' '

  # x2 = fh2.createVariable('mask_ml_sf',np.int32,('Nobs','pressure'),fill_value=0,zlib=True)
  # x2.long_name     = 'big mask with sequences of mixed layers in salt-finger regime'
  # x2.standard_name = 'mask_ml_sf'
  # x2.units         = ' '

  # x2 = fh2.createVariable('mask_int_dc_layer',np.int32,('Nobs','gradient_layers'),fill_value=0,zlib=True)
  # x2.long_name     = 'mask with sequences of interfaces in diffusive-convection regime'
  # x2.standard_name = 'mask_int_dc_layer'
  # x2.units         = ' '

  # x2 = fh2.createVariable('mask_ml_dc_layer',np.int32,('Nobs','mixed_layers'),fill_value=0,zlib=True)
  # x2.long_name     = 'mask with sequences of mixed layers in diffusive-convection regime'
  # x2.standard_name = 'mask_ml_dc_layer'
  # x2.units         = ' '

  # x2 = fh2.createVariable('mask_int_dc',np.int32,('Nobs','pressure'),fill_value=0,zlib=True)
  # x2.long_name     = 'big mask with sequences of interfaces in diffusive-convection regime'
  # x2.standard_name = 'mask_int_dc'
  # x2.units         = ' '

  # x2 = fh2.createVariable('mask_ml_dc',np.int32,('Nobs','pressure'),fill_value=0,zlib=True)
  # x2.long_name     = 'big mask with sequences of mixed layers in diffusive-convection regime'
  # x2.standard_name = 'mask_ml_dc'
  # x2.units         = ' '


