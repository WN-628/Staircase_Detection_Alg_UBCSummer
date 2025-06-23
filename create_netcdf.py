import netCDF4 as netcdf
import numpy as np

def create_netcdf(filename,nlevels):
  
  fh = netcdf.Dataset(filename, 'w', format='NETCDF4')
  fh.createDimension('Nobs', None)
  fh.createDimension('nlevels', nlevels)

  # 2D pressure per profile
  p_var = fh.createVariable('pressure',     'f4', ('Nobs','nlevels'),
                            fill_value=np.nan, zlib=True)
  p_var.long_name     = 'pressure'
  p_var.standard_name = 'pressure'
  p_var.units         = 'meter'

  # 2D CT, SA, masks on same axis
  ct_var = fh.createVariable('ct',    'f4', ('Nobs','nlevels'),
                            fill_value=np.nan, zlib=True)
  sa_var = fh.createVariable('sa',    'f4', ('Nobs','nlevels'), fill_value=np.nan, zlib=True)
  
  for name in ('mask_ml','mask_int','mask_cl','mask_sc'):
      mv = fh.createVariable(name, 'i1', ('Nobs','nlevels'),
                              fill_value=0, zlib=True)
      mv.long_name = name
      
  for name in ('cl_mushy', 'cl_supermushy'):
      mv = fh.createVariable(name, 'i1', ('Nobs','nlevels'),
                              fill_value=0, zlib=True)
      mv.long_name = name

  # 1D profile metadata
  x2 = fh.createVariable('prof',np.int32, ('Nobs'),fill_value=0,zlib=True)
  x2.long_name     = 'Profile number of float'
  x2.standard_name = 'prof'
  
  x2 = fh.createVariable('FloatID',np.int32, ('Nobs'),fill_value=0,zlib=True)
  x2.long_name     = 'Float ID'
  x2.standard_name = 'FloatID'
  
  x2 = fh.createVariable('dates','f8',('Nobs',),fill_value=0,zlib=True)
  x2.long_name     = 'Profile date'
  x2.standard_name = 'dates'
  x2.units         = 'seconds since 1970-01-01T00:00:00Z'
  x2.calendar      = 'gregorian'
  
  x2 = fh.createVariable('lon','f4',('Nobs',),fill_value=0,zlib=True)
  x2.long_name     = 'Longitude of float'
  x2.standard_name = 'lon'
  x2.units         = 'degrees'
  
  x2 = fh.createVariable('lat','f4',('Nobs',),fill_value=0,zlib=True)
  x2.long_name     = 'Latitude of float'
  x2.standard_name = 'lat'
  x2.units         = 'degrees'
  
  # min and max temperature
  x2 = fh.createVariable('depth_max_T', np.float32, ('Nobs',), fill_value=np.nan)
  x2.long_name = 'Depth of maximum temperature'
  x2.standard_name = 'max_depth_T'
  x2.units         = 'degrees Celsius'

  x2 = fh.createVariable('depth_min_T', np.float32, ('Nobs',), fill_value=np.nan)
  x2.long_name = 'Depth of minimum temperature'
  x2.standard_name = 'min_depth_T'
  x2.units         = 'degrees Celsius'
  
  return fh

