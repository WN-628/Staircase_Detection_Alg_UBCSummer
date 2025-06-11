import gsw
import netCDF4 as netcdf
import numpy as np
import scipy.interpolate as interpolate
import scipy.stats as stats
import scipy.ndimage as ndimage
import scipy.io
import pandas as pd
import ftplib as ftp
import wget 
import os
import h5py
import re

from config import FIXED_RESOLUTION_DBAR

def data_list_argo(centers,filename):
  #get list of documents
  list_ftp = ftp.FTP('ftp.ifremer.fr')
  list_ftp.login()
  list_ftp.cwd('ifremer/argo/dac/')
  for i in range(len(centers)):
    if i == 0:
      list_ftp.cwd(centers[i]+'/')
      floats    = list_ftp.nlst()
      directory = i*np.ones(len(list_ftp.nlst()),dtype=np.int32)
    else:
      list_ftp.cwd('../'+centers[i]+'/')
      floats    = np.concatenate((floats, list_ftp.nlst()),axis=0)
      directory = np.concatenate((directory,i*np.ones(len(list_ftp.nlst()),dtype=np.int32)),axis=0)
  floats = np.array(floats,dtype=np.int32)
  list_ftp.close()

  #download Greylist
  wget.download('ftp://ftp.ifremer.fr/ifremer/argo/ar_greylist.txt')
  grey_list   = pd.read_csv('ar_greylist.txt')
  greylist    = grey_list.PLATFORM_CODE.values
  os.remove('ar_greylist.txt')

  greylist = np.ma.concatenate((greylist,np.array([1900652,2900160,2901000,3900379,1900176,3900755,7900024,7900025,1900652,2900160,3900379,41534,4901506])),axis=0)

  float_list   = pd.read_excel(filename)
  floatlist    = np.unique(float_list.float_number.values)
  floatlist    = np.ma.array(floatlist,mask=np.zeros(len(floatlist)))
  floatlist.mask[np.argwhere(np.in1d(floatlist, np.intersect1d(floatlist,greylist)) == True)]=True

  float_list = floatlist[floatlist.mask==False]
  return directory, floats, float_list

def data_list_itp():
  #get list of documents of itp
  list_ftp = ftp.FTP('ftp.whoi.edu')
  list_ftp.login()
  list_ftp.prot_p() 
  list_ftp.cwd('/whoinet/itpdata/')

  floats_tmp = list_ftp.nlst()
  list_ftp.quit()
  
  for i in range(len(floats_tmp)):
    if floats_tmp[i][-9:] != 'final.zip':
      floats_tmp[i]= ''
  floats = np.unique(floats_tmp)
  float_list = floats[1:]
  return float_list

def load_data(filename,interp):
  fh    = netcdf.Dataset(filename,'r')
  if 'TEMP' in fh.variables and 'PSAL' in fh.variables and 'PRES' in fh.variables:
    #maximum depth > 1000 m
    """
    TO DO: #perform some simple quality checks
    """
    if interp == True:
      p0    = fh.variables['PRES'][:]
      lat   = fh.variables['LATITUDE'][:]
      lon   = fh.variables['LONGITUDE'][:]
      temp0 = fh.variables['TEMP'][:]
      salt0 = fh.variables['PSAL'][:]
      juld  = fh.variables['JULD'][:]
      diff = p0[:,1:]-p0[:,:-1]

      #use the adjusted profiles if available
      if 'TEMP_ADJUSTED' in fh.variables and 'PSAL_ADJUSTED' in fh.variables and 'PRES_ADJUSTED' in fh.variables:
        p_adjust =  fh.variables['PRES_ADJUSTED'][:]
        t_adjust = np.ma.masked_invalid(fh.variables['TEMP_ADJUSTED'][:])
        s_adjust = np.ma.masked_invalid(fh.variables['PSAL_ADJUSTED'][:])

        for f in range(len(p0[:,0])):
          if p_adjust[f,0]:
            p0[f,:]    = p_adjust[f,:]
            temp0[f,:] = t_adjust[f,:]
            salt0[f,:] = s_adjust[f,:]
        del p_adjust, t_adjust,s_adjust
      array_shape = np.ones([len(p0[:,0]),len(np.arange(0,2000))])
      temp   = np.ma.array(0*array_shape,mask=array_shape)
      salt   = np.ma.array(0*array_shape,mask=array_shape)
      ct     = np.ma.array(0*array_shape,mask=array_shape)
      sa     = np.ma.array(0*array_shape,mask=array_shape)
      p      = np.ma.array(0*array_shape,mask=array_shape)
      for i in range(len(p0[:,0])):
        if p0[i,:].min()>0 and p0[i,:].min()<20 and p0[i,:].max()>500:
          #surface value should between 0 and 20 dbar, maximum depth more than 1000m
#          index0 = np.where(p0[i,:]>1000)[0][0]
          steps = diff[i,:].mean()
          if steps < 5:
            #check if measurements are continuous
            mini = np.argmin(p0[i,:])
            maxi = np.argmax(p0[i,:])
            pres = p0[i,mini:maxi]
            t_tmp = temp0[i,mini:maxi]
            s_tmp = salt0[i,mini:maxi]
            if len(pres[pres.mask==True])>0 or len(t_tmp[t_tmp.mask==True])>0 or len(s_tmp[s_tmp.mask==True])>0:
             # remove all profiles with missing values
             # print('Not continuous')
              x = 1
            else:
              temp1 = interpolate.interp1d(p0[i,:],temp0[i,:])
              salt1 = interpolate.interp1d(p0[i,:],salt0[i,:])
              pmin  = int(np.ceil(p0[i,:].min()))
              pmax  = int(min(np.floor(p0[i,:].max()),1999))
              p[i,:] = np.arange(0,2000)
              p[i,:pmin].mask = True
              p[i,pmax:].mask = True
              temp[i,pmin:pmax] = temp1(np.arange(pmin,pmax))
              salt[i,pmin:pmax] = salt1(np.arange(pmin,pmax))

              if np.ma.abs(temp).max()<50:
                #convert to absolute salinity and conservative temperature
                sa[i,:]       = gsw.SA_from_SP(salt[i,:],p[i,:],lon[i],lat[i])
                ct[i,:]       = gsw.CT_from_t(sa[i,:],temp[i,:],p[i,:])
  else:
    p     = np.ma.array([0],mask=True)
    lat   = np.ma.array([0],mask=True)
    lon   = np.ma.array([0],mask=True)
    ct    = np.ma.array([0],mask=True)
    sa    = np.ma.array([0],mask=True)
    juld  = np.ma.array([0],mask=True)
  fh.close()
  return p,lat,lon,ct,sa,juld

def load_data_itp(path,profiles,interp=True, resolution = FIXED_RESOLUTION_DBAR):
  if profiles[0][-8:-4].isdigit():
    prof_no = np.array([int(profiles[0][-8:-4])])
    for no in np.arange(1,len(profiles)):
      if profiles[no][-8:-4].isdigit():
        prof_no = np.append(prof_no,int(profiles[no][-8:-4]))
      else:
         profiles[no] = 'no_data'
  else:
    if len(profiles)==1:
      profiles[0] = 'no_data'
      prof_no = [0]
    else:
      prof_no = np.array([int(profiles[1][-8:-4])])
      for no in np.arange(2,len(profiles)):
        if profiles[no][-8:-4].isdigit():
          prof_no = np.append(prof_no,int(profiles[no][-8:-4]))
        else:
          profiles[no] = 'no_data'
  x=-1
  for f in range(len(profiles)):
    if f == 0:
      target_levels = np.arange(0,2000,resolution)
      array_shape = np.ones([len(prof_no),len(target_levels)])
      lat    = np.ma.array(np.zeros(len(prof_no)),mask=np.ones(len(prof_no)))
      lon    = np.ma.array(np.zeros(len(prof_no)),mask=np.ones(len(prof_no)))
      juld    = np.ma.array(np.zeros(len(prof_no)),mask=np.ones(len(prof_no)))
      temp   = np.ma.array(0*array_shape,mask=array_shape)
      salt   = np.ma.array(0*array_shape,mask=array_shape)
      ct     = np.ma.array(0*array_shape,mask=array_shape)
      sa     = np.ma.array(0*array_shape,mask=array_shape)
      p      = np.ma.array(0*array_shape,mask=array_shape)
    if profiles[f] != 'no_data':
      x = x+1
      dat = pd.read_table(path+profiles[f],header=2,skipfooter=1,engine='python',delim_whitespace=True)
      if 'temperature(C)' and 'salinity' and '%pressure(dbar)' in dat.columns:
        if interp == True:
          temp0 = dat['temperature(C)'][:].values
          salt0 = dat['salinity'][:].values
          p0    = dat['%pressure(dbar)'][:].values

          dat2 = pd.read_table(path+profiles[f],engine='python',delim_whitespace=True,header=None,nrows=2).iloc[1]
          date = pd.Timestamp(year=int(dat2[0]),month=1,day=1)
          juld[x] = date.to_julian_date() - 1.5 + float(dat2[1])
          lon[x]  = float(dat2[2])
          lat[x]  = float(dat2[3])

          diff = p0[1:]-p0[:-1]
          if p0.min()>0 and p0.min()<20 and p0.max()>500:
            #surface value should between 0 and 20 dbar, maximum depth more than 1000m
  #          index0 = np.where(p0[i,:]>1000)[0][0]
            steps = diff.mean()
            if steps < 5:
              #check if measurements are continuous
              mini = np.argmin(p0)
              maxi = np.argmax(p0)
              pres = p0[mini:maxi]
              t_tmp = temp0[mini:maxi]
              s_tmp = salt0[mini:maxi]
              temp1 = interpolate.interp1d(p0,temp0)
              salt1 = interpolate.interp1d(p0,salt0)
              pmin  = int(np.ceil(p0.min()))
              pmax  = int(min(np.floor(p0.max()),1999))
              p[x,:] = np.arange(0,2000)
              p[x,:pmin].mask = True
              p[x,pmax:].mask = True
              temp[x,pmin:pmax] = temp1(np.arange(pmin,pmax))
              salt[x,pmin:pmax] = salt1(np.arange(pmin,pmax))

              #convert to absolute salinity and conservative temperature
              sa[x,:]       = gsw.SA_from_SP(salt[x,:],p[x,:],lon[x],lat[x])
              ct[x,:]       = gsw.CT_from_t(sa[x,:],temp[x,:],p[x,:])

  return prof_no,p,lat,lon,ct,sa,juld




def load_data_mat_zip(path, profiles, interp=True, resolution=FIXED_RESOLUTION_DBAR):
    valid_profiles = []
    target_levels = np.arange(0, 2000, resolution)
    # print("Files in tmp/:", profiles)

    for fname in profiles:
        if not fname.endswith(".mat"):
            continue
        try:
            with h5py.File(os.path.join(path, fname), 'r') as f:
                pressure = f['pr_filt'][:].flatten()
                temperature = f['te_cor'][:].flatten()
                salinity = f['sa_cor'][:].flatten()
                lat = float(f['latitude'][0][0])
                lon = float(f['longitude'][0][0])
        except Exception as e:
            # print(f"Skipping {fname}: {e}")
            continue

        # print(f"{fname}: pressure min = {pressure.min()}, max = {pressure.max()}, steps = {np.mean(np.diff(pressure))}")

        if pressure.min() > 0 and pressure.max() > 500:
            p_interp = target_levels
            try:
                temp_interp = np.interp(p_interp, pressure, temperature)
                salt_interp = np.interp(p_interp, pressure, salinity)
            except Exception as e:
                # print(f"Interpolation failed for {fname}: {e}")
                continue

            sa = gsw.SA_from_SP(salt_interp, p_interp, lon, lat)
            ct = gsw.CT_from_t(sa, temp_interp, p_interp)

            profile = {
                "p": p_interp,
                "ct": ct,
                "sa": sa,
                "lat": lat,
                "lon": lon,
                "juld": 0,  # default value (not found in .mat)
                "prof_no": int(fname[-8:-4]) if fname[-8:-4].isdigit() else 0
            }
            valid_profiles.append(profile)

    N = len(valid_profiles)
    array_shape = (N, len(target_levels))

    p = np.ma.masked_all(array_shape)
    ct = np.ma.masked_all(array_shape)
    sa = np.ma.masked_all(array_shape)
    lat = np.ma.masked_all(N)
    lon = np.ma.masked_all(N)
    juld = np.ma.masked_all(N)
    prof_no = np.zeros(N, dtype=int)

    for i, prof in enumerate(valid_profiles):
        p[i, :] = prof['p']
        ct[i, :] = prof['ct']
        sa[i, :] = prof['sa']
        lat[i] = prof['lat']
        lon[i] = prof['lon']
        juld[i] = prof['juld']
        prof_no[i] = prof['prof_no']
        
    # print(f"✅ Loaded {len(valid_profiles)} valid profile(s): {[p['prof_no'] for p in valid_profiles]}")

    return prof_no, p, lat, lon, ct, sa, juld
  
  
  
def load_data_csv_zip(path, profiles, interp=True, resolution=FIXED_RESOLUTION_DBAR):
    valid_profiles = []
    target_levels = np.arange(0, 2000, resolution)

    for fname in profiles:
        if not fname.endswith(".csv") or os.path.basename(fname).startswith("._"):
            continue

        full_path = os.path.join(path, fname)
        try:
            df = pd.read_csv(full_path)
            df.columns = df.columns.str.strip().str.lower()
            pressure = df['depth'].to_numpy().flatten() if 'depth' in df.columns else df.iloc[:, 0].to_numpy().flatten()
            temperature = df['temperature'].to_numpy().flatten() if 'temperature' in df.columns else df.iloc[:, 1].to_numpy().flatten()
            salinity = df['salinity'].to_numpy().flatten() if 'salinity' in df.columns else df.iloc[:, 2].to_numpy().flatten()
        except Exception as e:
            print(f"❌ Failed to read {fname}: {e}")
            continue

        # Extract lat/lon from filename
        try:
            base = os.path.basename(fname)
            parts = base.split('_')
            lon = float(parts[1])
            lat = float(parts[2])
        except Exception as e:
            print(f"⚠️ Failed to extract lat/lon from {fname}: {e}")
            lat, lon = 0.0, 0.0

        if pressure.size == 0 or pressure.max() <= 500:
            print(f"⛔ Skipping {fname}: invalid pressure range")
            continue

        try:
            if interp:
                max_p = pressure.max()
                p_interp = np.arange(0, max_p + resolution, resolution)
                temp_interp = np.interp(p_interp, pressure, temperature)
                salt_interp = np.interp(p_interp, pressure, salinity)

                sa = gsw.SA_from_SP(salt_interp, p_interp, lon, lat)
                ct = gsw.CT_from_t(sa, temp_interp, p_interp)

                profile = {
                    "p": p_interp,
                    "ct": ct,
                    "sa": sa,
                }
            else:
                sa = gsw.SA_from_SP(salinity, pressure, lon, lat)
                ct = gsw.CT_from_t(sa, temperature, pressure)
                profile = {
                    "p": pressure,
                    "ct": ct,
                    "sa": sa,
                }

            match = re.search(r"cor(\d{4})_", base)
            prof_number = int(match.group(1)) if match else 0
            profile.update({
                "lat": lat,
                "lon": lon,
                "juld": 0,
                "prof_no": prof_number
            })
            valid_profiles.append(profile)
        except Exception as e:
            print(f"⚠️ GSW conversion failed for {fname}: {e}")
            continue

    N = len(valid_profiles)
    if N == 0:
        print("⛔ No valid profiles found.")
        return [], None, None, None, None, None, None
    
    N = len(valid_profiles)
    array_shape = (N, len(target_levels))

    p = np.ma.masked_all(array_shape)
    ct = np.ma.masked_all(array_shape)
    sa = np.ma.masked_all(array_shape)
    lat = np.ma.masked_all(N)
    lon = np.ma.masked_all(N)
    juld = np.ma.masked_all(N)
    prof_no = np.zeros(N, dtype=int)

    # max_len = max(len(p["p"]) for p in valid_profiles)

    # p = np.ma.masked_all((N, max_len))
    # ct = np.ma.masked_all((N, max_len))
    # sa = np.ma.masked_all((N, max_len))
    # lat = np.ma.masked_all(N)
    # lon = np.ma.masked_all(N)
    # juld = np.ma.masked_all(N)
    # prof_no = np.zeros(N, dtype=int)

    for i, prof in enumerate(valid_profiles):
        L = len(prof["p"])
        p[i, :L] = prof["p"]
        ct[i, :L] = prof["ct"]
        sa[i, :L] = prof["sa"]
        lat[i] = prof["lat"]
        lon[i] = prof["lon"]
        juld[i] = prof["juld"]
        prof_no[i] = prof["prof_no"]

    print(f"✅ Loaded {N} valid profile(s)")
    return prof_no, p, lat, lon, ct, sa, juld
