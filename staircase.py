import numpy as np
from types import SimpleNamespace
from scipy.ndimage import uniform_filter1d

def cent_derivative2d(f,z):
  dz1     = z[:,1:-1]-z[:,:-2]
  dz2     = z[:,2:]-z[:,1:-1]
  fkp1    = f[:,2:]
  fkm1    = f[:,:-2]
  fk      = f[:,1:-1]

  # dfdz    = np.ma.zeros(f.shape)
  dfdz = np.ma.array(np.zeros(f.shape), mask=np.zeros(f.shape, dtype=bool))
  dfdz[:,1:-1] = ( dz1**2*fkp1 + ( dz2**2-dz1**2 )*fk - dz2**2*fkm1
                ) / ( dz1*dz2*(dz1+dz2) )
  dfdz.mask[dfdz==0]=True 
  return dfdz

def detect_mixed_layers_dual(p, ct,
                             smooth_window=5,
                             mixed_layer_threshold=0.0002,
                             interface_threshold=0.005):
    """
    Detect mixed layers using smoothed CT, but detect interfaces on the raw CT.

    Parameters
    ----------
    p : 2D array (n_profiles, n_levels)
        Pressure or depth.
    ct : 2D array, same shape as p
        Conservative temperature.
    smooth_window : int
        Width of moving‐average smoother (odd).
    mixed_layer_threshold : float
        |dCT_smooth/dp| below → mixed layer.
    interface_threshold : float
        |dCT_raw/dp| above → interface.

    Returns
    -------
    mask_ml  : 2D bool array
        True where a mixed layer is detected (on smoothed CT).
    mask_int : 2D bool array
        True where an interface is detected (on raw CT).
    """
    n_prof, n_lev = ct.shape
    # 1) Smooth each profile (you can swap in gaussian or Savitzky–Golay)
    half_w = smooth_window // 2
    kernel = np.ones(smooth_window) / smooth_window
    ct_smooth = np.zeros_like(ct)
    for i in range(n_prof):
        prof = np.ma.getdata(ct[i])
        padded = np.pad(prof, pad_width=half_w, mode='edge')
        ct_smooth[i] = np.convolve(padded, kernel, mode='valid')

    # 2) Initialize masks
    mask_ml  = np.zeros_like(ct, dtype=bool)
    mask_int = np.zeros_like(ct, dtype=bool)

    # 3) Loop over profiles & levels
    for i in range(n_prof):
        for j in range(1, n_lev - 1):
            # skip masked
            if np.ma.is_masked(ct[i, j-1]) or np.ma.is_masked(ct[i, j+1]):
                continue

            dp = p[i, j+1] - p[i, j-1]
            if dp == 0:
                continue

            # raw CT gradient for interface
            grad_raw   = (ct[i, j+1]   - ct[i, j-1])   / dp
            # smoothed CT gradient for mixed layer
            grad_smooth = (ct_smooth[i, j+1] - ct_smooth[i, j-1]) / dp

            if abs(grad_smooth) < mixed_layer_threshold:
                mask_ml[i, j] = True
            if abs(grad_raw) > interface_threshold:
                mask_int[i, j] = True

    return mask_ml, mask_int



def mask_continuity(mask, length):
    """
    For each row in the 2D boolean array `mask`, find all contiguous runs of True.
    If a run’s length is < `length`, set those positions to False.

    Args:
        mask : 2D boolean array of shape (n_profiles, n_levels).
        length : int
            Minimum run‐length (in grid points) to retain.

    Returns:
        2D boolean array of the same shape, with all runs shorter than `length` zeroed out.
    """
    # Convert masked arrays to plain booleans
    if isinstance(mask, np.ma.MaskedArray):
        m = mask.filled(False)
    else:
        m = np.asarray(mask, dtype=bool)

    out = m.copy()
    n_rows, n_cols = m.shape

    for i in range(n_rows):
        row = m[i]
        j = 0
        while j < n_cols:
            if row[j]:
                start = j
                # advance until the run ends
                while j < n_cols and row[j]:
                    j += 1
                run_len = j - start
                if run_len < length:
                    # zero‐out any too‐short runs
                    out[i, start:j] = False
            else:
                j += 1

    return out

def continuity(arr, num_one, num_two, num_three):
    """
    1) Remove runs of 1,2,3 shorter than thresholds.
    2) Bridge short zero-gaps (<= num_three) by marking as 3.
    3) Keep only blocks with ≥3 alternating 1↔2 runs.
    """
    out = list(arr)
    n = len(out)

    # Stage 1: drop too-short runs
    for val, thresh in ((1, num_one), (2, num_two), (3, num_three)):
        i = 0
        while i < n:
            if out[i] == val:
                start = i
                while i < n and out[i] == val:
                    i += 1
                if (i - start) < thresh:
                    for j in range(start, i):
                        out[j] = 0
            else:
                i += 1

    # Stage 2: fill short zero-gaps with 3
    i = 0
    while i < n:
        if out[i] != 0:
            i += 1
            continue
        start = i
        while i < n and out[i] == 0:
            i += 1
        end = i
        run_len = end - start
        if start > 0 and end < n and out[start-1] != 0 and out[end] != 0 and run_len <= num_three:
            for j in range(start, end):
                out[j] = 3

    # Stage 3: alternation test
    i = 0
    while i < n:
        if out[i] == 0:
            i += 1
            continue
        rs = i
        while i < n and out[i] != 0:
            i += 1
        re = i
        runs = []
        j = rs
        while j < re:
            if out[j] in (1, 2):
                v = out[j]
                while j < re and out[j] == v:
                    j += 1
                runs.append(v)
            else:
                j += 1
        ok = False
        for k in range(len(runs)):
            cnt, last = 1, runs[k]
            for l in range(k+1, len(runs)):
                if runs[l] != last:
                    cnt += 1
                    last = runs[l]
                else:
                    break
            if cnt >= 3:
                ok = True
                break
        if not ok:
            for j in range(rs, re):
                out[j] = 0
    return np.array(out)

def get_mixed_layers(p, ct,
                    thres_ml=0.005,
                    thres_int=0.005,
                    min_run=3,
                    mushy=1.0):
    '''
    Identify mixed layers, interfaces, and staircase structures.

    Args:
        p         : 2D array, shape (n_profiles, n_levels), increasing downward (dbar or m)
        ct        : 2D array, same shape, conservative temperature
        thres_ml  : slope threshold for mixed-layer points
        thres_int  : slope threshold for gradient-layer points
        min_run   : minimum contiguous points per ml/int run (grid points)
        mushy     : max vertical separation (same units as p) to link runs

    Returns:
        masks.ml  : boolean mask of mixed-layer points
        masks.int  : boolean mask of gradient-layer points
        masks.cl  : boolean mask of connection (link) points
        masks.sc  : boolean mask of all staircase structure points
    '''
    
    # 0. find max‐T and first local min depths per profile
    n_prof, n_lev = ct.shape
    depth_max_T = np.full(n_prof, np.nan)
    depth_min_T = np.full(n_prof, np.nan)
    for k in range(n_prof):
        temp = np.ma.masked_invalid(ct[k])
        pres = np.ma.masked_where(temp.mask, p[k])
        if temp.count() == 0:
            continue
        depth_max_T[k] = pres[np.ma.argmax(temp)]
        depth_min_T[k] = pres[np.ma.argmin(temp)]

    # 0A. restrict to ±25 m around those extrema :contentReference[oaicite:1]{index=1}
    valid_depth_mask = np.zeros_like(p, dtype=bool)
    for k in range(n_prof):
        dmin = depth_min_T[k] + 25
        dmax = depth_max_T[k] - 25
        valid_depth_mask[k] = (p[k] >= dmin) & (p[k] <= dmax)

    # mask out everything outside that window
    ct = np.ma.masked_where(~valid_depth_mask, ct)
    p  = np.ma.masked_where(~valid_depth_mask, p)
    
    assert np.shape(ct) == np.shape(p), "ct and p must have the same shape after checking max and min temperature."
    
    # # 1. compute vertical derivative dT/dz (Kat et al.’s method) :contentReference[oaicite:2]{index=2}
    # dTdz = cent_derivative2d(ct, p)

    # # 2. initial masks by gradient thresholds
    # mask_ml = np.abs(dTdz) < thres_ml
    # mask_int = np.abs(dTdz) > thres_int
    # assert np.any(mask_ml), "No mixed layer mask found in gradient."
    # assert np.any(mask_int), "No interface mask found in gradient."
    
    mask_ml, mask_int = detect_mixed_layers_dual(p, ct, 11, thres_ml, thres_int)

    # 3. prune short runs (thickness condition) :contentReference[oaicite:3]{index=3}
    # mask_ml = mask_continuity(mask_ml, min_run)
    # mask_int = mask_continuity(mask_int, min_run)
    assert np.any(mask_ml), "No mixed layer mask found in the data with gradient check."
    
    # 3. assemble staircase structure
    clean_ml  = np.zeros_like(mask_ml,  dtype=bool)
    clean_int = np.zeros_like(mask_int, dtype=bool)
    mask_cl   = np.zeros_like(mask_ml,  dtype=bool)
    mask_sc   = np.zeros_like(mask_ml,  dtype=bool)

    resolution   = 0.25
    mushy_points = int(np.ceil(mushy / resolution))

    for i in range(n_prof):
        arr = np.zeros(n_lev, dtype=int)
        arr[mask_ml[i]]  = 1
        arr[mask_int[i]] = 2

        cleaned = continuity(arr, min_run, min_run, mushy_points)
        clean_ml[i]  = (cleaned == 1)
        clean_int[i] = (cleaned == 2)
        mask_cl[i]   = (cleaned == 3)
        mask_sc[i]   = (cleaned > 0)
    
    assert np.any(clean_int), "No interface mask found in the data with continuity check."
    assert np.any(clean_ml), "No mixed layer mask found in the data with continuity check."

    masks = SimpleNamespace(ml=clean_ml, int=clean_int,
                                cl=mask_cl, sc=mask_sc)
    return masks, depth_min_T, depth_max_T