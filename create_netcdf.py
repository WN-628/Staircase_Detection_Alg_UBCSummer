import netCDF4 as netcdf
import numpy as np

from config import FIXED_RESOLUTION_DBAR

def create_netcdf(filename,max_count):
  fh2 = netcdf.Dataset(filename,'w',format='NETCDF4')
  fh2.createDimension('Nobs',None)
  x1 = fh2.createVariable('n', np.int32, ('Nobs'))
  x1.long_name     = 'Profile'
  x1.standard_name = 'no'
  
  pressure_levels = np.arange(0, 2000, FIXED_RESOLUTION_DBAR)
  
  fh2.createDimension('pressure', len(pressure_levels))
  x1 = fh2.createVariable('pressure', 'f4', ('pressure'))
  x1.long_name     = 'Pressure'
  x1.standard_name = 'depth'
  x1.units         = 'dbar'
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
  
  x2 = fh2.createVariable('juld','f4',('Nobs'),fill_value=0,zlib=True)
  x2.long_name     = 'Julian date of profile'
  x2.standard_name = 'juld'
  x2.units         = 'days after 1950-01-01'
  
  x2 = fh2.createVariable('lon','f4',('Nobs'),fill_value=0,zlib=True)
  x2.long_name     = 'Longitude of float'
  x2.standard_name = 'lon'
  x2.units         = 'degrees'
  
  x2 = fh2.createVariable('lat','f4',('Nobs'),fill_value=0,zlib=True)
  x2.long_name     = 'Latitude of float'
  x2.standard_name = 'lat'
  x2.units         = 'degrees'

  # other variables
  x2 = fh2.createVariable('ct','f4',('Nobs','pressure'),fill_value=0,zlib=True)
  x2.long_name     = 'Conservative Temperature'
  x2.standard_name = 'ct'
  x2.units         = 'degrees Celsius'

  x2 = fh2.createVariable('sa','f4',('Nobs','pressure'),fill_value=0,zlib=True)
  x2.long_name     = 'Absolute Salinity'
  x2.standard_name = 'sa'
  x2.units         = 'g/kg'

  # mixed layers
  x2 = fh2.createVariable('ml_T','f4',('Nobs','mixed_layers'),fill_value=0,zlib=True)
  x2.long_name     = 'average temperature of the mixed layer'
  x2.standard_name = 'ml_T'
  x2.units         = 'degrees Celsius'
 
  x2 = fh2.createVariable('ml_S','f4',('Nobs','mixed_layers'),fill_value=0,zlib=True)
  x2.long_name     = 'average salinity of the mixed layer'
  x2.standard_name = 'ml_S'
  x2.units         = 'g kg-1'
 
  x2 = fh2.createVariable('ml_r','f4',('Nobs','mixed_layers'),fill_value=0,zlib=True)
  x2.long_name     = 'average density of the mixed layer (sigma1)'
  x2.standard_name = 'ml_r'
  x2.units         = 'kg m-3'

  x2 = fh2.createVariable('ml_p','f4',('Nobs','mixed_layers'),fill_value=0,zlib=True)
  x2.long_name     = 'average pressure of the mixed layer'
  x2.standard_name = 'ml_p'
  x2.units         = 'dbar'

  x2 = fh2.createVariable('ml_h','f4',('Nobs','mixed_layers'),fill_value=0,zlib=True)
  x2.long_name     = 'height of the mixed layer'
  x2.standard_name = 'ml_h'
  x2.units         = 'dbar'
 
  x2 = fh2.createVariable('ml_Tu','f4',('Nobs','mixed_layers'),fill_value=0,zlib=True)
  x2.long_name     = 'average Turner angle of the mixed layer'
  x2.standard_name = 'ml_Tu'
  x2.units         = 'degrees'

  x2 = fh2.createVariable('ml_R','f4',('Nobs','mixed_layers'),fill_value=0,zlib=True)
  x2.long_name     = 'average density ratio of the mixed layer'
  x2.standard_name = 'ml_R'
  x2.units         = ' '

  #gradient layer variables
  x2 = fh2.createVariable('gl_dT','f4',('Nobs','gradient_layers'),fill_value=0,zlib=True)
  x2.long_name     = 'temperature difference in gradient layer'
  x2.standard_name = 'gl_dT'
  x2.units         = 'degrees Celsius'

  x2 = fh2.createVariable('gl_dS','f4',('Nobs','gradient_layers'),fill_value=0,zlib=True)
  x2.long_name     = 'salinity difference in gradient layer'
  x2.standard_name = 'gl_dS'
  x2.units         = 'g kg-1'

  x2 = fh2.createVariable('gl_dr','f4',('Nobs','gradient_layers'),fill_value=0,zlib=True)
  x2.long_name     = 'density difference in gradient layer (sigma1)'
  x2.standard_name = 'gl_dr'
  x2.units         = 'kg m-3 dbar-1'

  x2 = fh2.createVariable('gl_h','f4',('Nobs','gradient_layers'),fill_value=0,zlib=True)
  x2.long_name     = 'gradient layer height'
  x2.standard_name = 'gl_h'
  x2.units         = 'dbar'

  x2 = fh2.createVariable('gl_Tu','f4',('Nobs','gradient_layers'),fill_value=0,zlib=True)
  x2.long_name     = 'average Turner angle of the gradient layer'
  x2.standard_name = 'gl_Tu'
  x2.units         = 'degrees'

  x2 = fh2.createVariable('gl_R','f4',('Nobs','gradient_layers'),fill_value=0,zlib=True)
  x2.long_name     = 'average density ratio of the gradient layer'
  x2.standard_name = 'gl_R'
  x2.units         = ' '

  x2 = fh2.createVariable('mask_gl_sf_layer',np.int32,('Nobs','gradient_layers'),fill_value=0,zlib=True)
  x2.long_name     = 'mask with sequences of gradient layers in salt-finger regime'
  x2.standard_name = 'mask_gl_sf_layer'
  x2.units         = ' '

  x2 = fh2.createVariable('mask_ml_sf_layer',np.int32,('Nobs','mixed_layers'),fill_value=0,zlib=True)
  x2.long_name     = 'mask with sequences of mixed layers in salt-finger regime'
  x2.standard_name = 'mask_ml_sf_layer'
  x2.units         = ' '

  x2 = fh2.createVariable('mask_gl_sf',np.int32,('Nobs','pressure'),fill_value=0,zlib=True)
  x2.long_name     = 'big mask with sequences of gradient layers in salt-finger regime'
  x2.standard_name = 'mask_gl_sf'
  x2.units         = ' '

  x2 = fh2.createVariable('mask_ml_sf',np.int32,('Nobs','pressure'),fill_value=0,zlib=True)
  x2.long_name     = 'big mask with sequences of mixed layers in salt-finger regime'
  x2.standard_name = 'mask_ml_sf'
  x2.units         = ' '

  x2 = fh2.createVariable('mask_gl_dc_layer',np.int32,('Nobs','gradient_layers'),fill_value=0,zlib=True)
  x2.long_name     = 'mask with sequences of gradient layers in diffusive-convection regime'
  x2.standard_name = 'mask_gl_dc_layer'
  x2.units         = ' '

  x2 = fh2.createVariable('mask_ml_dc_layer',np.int32,('Nobs','mixed_layers'),fill_value=0,zlib=True)
  x2.long_name     = 'mask with sequences of mixed layers in diffusive-convection regime'
  x2.standard_name = 'mask_ml_dc_layer'
  x2.units         = ' '

  x2 = fh2.createVariable('mask_gl_dc',np.int32,('Nobs','pressure'),fill_value=0,zlib=True)
  x2.long_name     = 'big mask with sequences of gradient layers in diffusive-convection regime'
  x2.standard_name = 'mask_gl_dc'
  x2.units         = ' '

  x2 = fh2.createVariable('mask_ml_dc',np.int32,('Nobs','pressure'),fill_value=0,zlib=True)
  x2.long_name     = 'big mask with sequences of mixed layers in diffusive-convection regime'
  x2.standard_name = 'mask_ml_dc'
  x2.units         = ' '

  fh2.close()
