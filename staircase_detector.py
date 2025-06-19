import numpy as np
from types import SimpleNamespace
from scipy.ndimage import uniform_filter1d
from config import FIXED_RESOLUTION_METER

def continuity(arr, num_one, num_three):
    out = list(arr)
    n = len(out)

    # Stage 1
    for val, thresh in ((1, num_one), (3, num_three)):
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

    # Stage 2
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

        if (start > 0 and end < n
            and out[start-1] != 0
            and out[end] != 0
            and out[start-1] != out[end]
            and run_len <= num_three):
            for j in range(start, end):
                out[j] = 3

    # Stage 3
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

def fill_gaps(mask_ml, mask_int, gap_ml, gap_int):
    """
    Fill short gaps in mixed‐layer and interface masks.

    Args:
        mask_ml : 2D bool array, shape (n_profiles, n_levels)
                    initial mixed‐layer mask
        mask_int: 2D bool array, same shape
                    initial interface mask
        gap_ml  : int
                    maximum number of consecutive False values in mask_ml
                    to fill between two True regions
        gap_int : int
                    maximum number of consecutive False values in mask_int
                    to fill between two True regions

    Returns:
        filled_ml, filled_int : two 2D bool arrays of same shape,
                    with short gaps filled in each mask
    """
    # Make copies so we don’t overwrite originals
    filled_ml  = mask_ml.copy()
    filled_int = mask_int.copy()
    
    n_prof, n_lev = mask_ml.shape
    
    for mask, filled, gap_thr in (
        (mask_ml, filled_ml,  gap_ml),
        (mask_int, filled_int, gap_int)
    ):
        for i in range(n_prof):
            j = 0
            while j < n_lev:
                # look for the start of a False‐run
                if not filled[i, j]:
                    start = j
                    while j < n_lev and not filled[i, j]:
                        j += 1
                    end = j  # first index after the False‐run
                    
                    # only fill if it’s strictly between two True‐values
                    if start > 0 and end < n_lev \
                        and filled[i, start-1] and filled[i, end]:
                        
                        run_length = end - start
                        if run_length <= gap_thr:
                            filled[i, start:end] = True
                else:
                    j += 1

    return filled_ml, filled_int

def interface_con(ct, mask_int, thres_temp=0.05):
    """
    Prune interfaces in a temperature profile based on a threshold.

    Parameters:
    - ct: 1D or 2D array of conservative temperature (°C)
    - mask_int: boolean mask array (same shape as ct) indicating initial interface detections
    - thres_temp: temperature difference threshold to prune narrow interfaces

    Returns:
    - mask_pruned: boolean mask with pruned interfaces
    """
    # Ensure inputs are at least 2D (profiles × levels)
    ct_arr = np.atleast_2d(ct)
    mask_arr = np.atleast_2d(mask_int)
    n_prof, n_lev = ct_arr.shape

    # Copy mask for pruning
    mask_pruned = mask_arr.copy()
    single_profile = (ct.ndim == 1)

    for i in range(n_prof):
        in_int = False
        start = None
        for j in range(n_lev):
            if mask_arr[i, j] and not in_int:
                # entering an interface
                in_int = True
                start = j
            elif in_int and not mask_arr[i, j]:
                # exiting an interface
                end = j
                delta_t = abs(ct_arr[i, end - 1] - ct_arr[i, start])
                if delta_t < thres_temp:
                    # prune: set interface points to False
                    mask_pruned[i, start:end] = False
                in_int = False
        # handle case where interface continues to end of profile
        if in_int and start is not None:
            delta_t = abs(ct_arr[i, -1] - ct_arr[i, start])
            if delta_t < thres_temp:
                mask_pruned[i, start:n_lev] = False

    # Return in original dimension
    if single_profile:
        return mask_pruned[0]
    return mask_pruned

def mixed_con(ct, mask_ml, thres_temp=0.05):
    """
    Prune interfaces in a temperature profile based on a threshold.

    Parameters:
    - ct: 1D or 2D array of conservative temperature (°C)
    - mask_int: boolean mask array (same shape as ct) indicating initial interface detections
    - thres_temp: temperature difference threshold to prune narrow interfaces

    Returns:
    - mask_pruned: boolean mask with pruned interfaces
    """
    # Ensure inputs are at least 2D (profiles × levels)
    ct_arr = np.atleast_2d(ct)
    mask_arr = np.atleast_2d(mask_ml)
    n_prof, n_lev = ct_arr.shape

    # Copy mask for pruning
    mask_pruned = mask_arr.copy()
    single_profile = (ct.ndim == 1)

    for i in range(n_prof):
        in_int = False
        start = None
        for j in range(n_lev):
            if mask_arr[i, j] and not in_int:
                # entering an interface
                in_int = True
                start = j
            elif in_int and not mask_arr[i, j]:
                # exiting an interface
                end = j
                delta_t = abs(ct_arr[i, end - 1] - ct_arr[i, start])
                if delta_t > thres_temp:
                    # prune: set interface points to False
                    mask_pruned[i, start:end] = False
                in_int = False
        # handle case where interface continues to end of profile
        if in_int and start is not None:
            delta_t = abs(ct_arr[i, -1] - ct_arr[i, start])
            if delta_t > thres_temp:
                mask_pruned[i, start:n_lev] = False

    # Return in original dimension
    if single_profile:
        return mask_pruned[0]
    return mask_pruned

def detect_layer_lengths(mask, p):
    """
    Compute the lengths of each contiguous True “layer” in a mask.

    Parameters
    ----------
    mask : ndarray of bool, shape (n_profiles, n_levels) or (n_levels,)
        Boolean mask array where True indicates the presence of a layer (mixed or interface).
    p : ndarray of float, same shape as mask or shape (n_levels,)
        Depth or pressure values corresponding to each level; must be increasing downward.

    Returns
    -------
    runs : list
        If input is 2D (n_profiles × n_levels), returns a list of length n_profiles. Each element is
        a dict with keys:
            'start_idx' : ndarray of int, start indices of each run
            'end_idx'   : ndarray of int, end   indices of each run
            'count'     : ndarray of int, number of grid points in each run
            'thickness' : ndarray of float, physical thickness = p[end] - p[start] for each run

        If input is 1D, returns a single dict (not wrapped in a list).

    Example
    -------
    >>> # single profile
    >>> mask = np.array([False, True, True, False, True])
    >>> p    = np.array([0., 1., 2., 3., 4.])
    >>> detect_layer_lengths(mask, p)
    {
        'start_idx': array([1, 4]),
        'end_idx':   array([2, 4]),
        'count':     array([2, 1]),
        'thickness': array([1., 0.])
    }
    """
    # Ensure 2D
    was1d = False
    m = mask
    if mask.ndim == 1:
        was1d = True
        m = mask[np.newaxis, :]
        # if p is 1D, make it 2D broadcastable
        if p.ndim == 1:
            p = np.tile(p, (1, 1))

    n_prof, n_lev = m.shape
    results = []

    for i in range(n_prof):
        row = m[i]
        starts = []
        ends   = []
        counts = []
        thick  = []

        j = 0
        while j < n_lev:
            if row[j]:
                start = j
                while j < n_lev and row[j]:
                    j += 1
                end = j - 1

                starts.append(start)
                ends.append(end)

                cnt = end - start + 1
                counts.append(cnt)

                # physical thickness
                # handle both 1D and 2D p
                pi = p[i] if p.ndim == 2 else p
                thickness = pi[end] - pi[start]
                thick.append(thickness)
            else:
                j += 1

        results.append({
            'start_idx': np.array(starts, dtype=int),
            'end_idx':   np.array(ends,   dtype=int),
            'count':     np.array(counts, dtype=int),
            'thickness': np.array(thick,  dtype=float)
        })

    return results[0] if was1d else results

def interface_height(p, mask_int, thres_height):
    """
    Mask out any interface whose physical thickness exceeds thres_height.

    Parameters
    ----------
    mask_int : ndarray of bool, shape (n_profiles, n_levels) or (n_levels,)
        Boolean mask where True marks an interface.
    p : ndarray of float, same shape as mask_int or shape (n_levels,)
        Depth or pressure values for each level (must be increasing).
    thres_height : float
        Maximum allowed interface thickness. Any contiguous run of True in
        mask_int whose thickness (p[end] - p[start]) > thres_height
        will be set to False.

    Returns
    -------
    filtered : ndarray of bool, same shape as mask_int
        A copy of mask_int with “too tall” interfaces removed.
    """
    # copy input mask
    filtered = mask_int.copy()
    # get runs of True
    runs = detect_layer_lengths(mask_int, p)
    # wrap single-profile output in list for uniformity
    is_1d = (mask_int.ndim == 1)
    if is_1d:
        runs = [runs]
        filtered = filtered[np.newaxis, :]
        p_arr = p[np.newaxis, :]
    else:
        p_arr = p

    # loop over each profile
    for i, run in enumerate(runs):
        for start, end, thickness in zip(run['start_idx'],
                                         run['end_idx'],
                                         run['thickness']):
            if thickness > thres_height:
                # mask out this entire run
                filtered[i, start:end+1] = False

    # if original was 1D, squeeze back
    return filtered[0] if is_1d else filtered

def detect_mixed_layers_dual(p, ct, smooth_window,
                            mixed_layer_threshold,
                            interface_threshold):
    """
    Detect mixed layers using smoothed CT, but detect interfaces on the raw CT based on temperature gradient.

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
            grad_raw   = (ct[i, j+1] - ct[i, j-1])   / dp
            # smoothed CT gradient for mixed layer
            grad_smooth = (ct_smooth[i, j+1] - ct_smooth[i, j-1]) / dp

            if abs(grad_smooth) < mixed_layer_threshold:
                mask_ml[i, j] = True
            if grad_raw > interface_threshold:
            # this cannot use absolute value, since we may discover mixed layer going backwards
                mask_int[i, j] = True

    return mask_ml, mask_int

def get_mixed_layers(p, ct,
                    thres_ml,
                    thres_int,
                    ml_min_depth=0.75, int_temp=0.01,
                    cl_length=1.0, smooth=11):
    '''
    Identify mixed layers, interfaces, and staircase structures for data in `p` and `ct`.

    Args:
        p         : 2D array, shape (n_profiles, n_levels), increasing downward (dbar or m)
        ct        : 2D array, same shape, conservative temperature
        thres_ml  : slope threshold for mixed-layer points
        thres_int : slope threshold for gradient-layer points
        min_run   : minimum contiguous points per ml/int run (grid points)
        mushy     : max vertical separation (same units as p) to link runs
        smooth    : width of moving‐average smoother (must be odd, default 11)

    Returns:
        masks.ml  : boolean mask of mixed-layer points
        masks.int : boolean mask of gradient-layer points
        masks.cl  : boolean mask of connection (link) points
        masks.sc  : boolean mask of all staircase structure points
    
    Note: All the returned masks are 2D arrays of the same shape as `p` and `ct`. 
    '''
    
    # Check input shapes
    assert np.shape(p) == np.shape(ct), "p and ct must have the same shape."
    
    # Parameters:
    gap_ml = 2  # maximum number of consecutive False values in mask_ml to fill
    gap_int = 1  # maximum number of consecutive False values in mask_int to fill
    ml_temp = 0.005  # maximum threshold for mixed layer temperature width
    ml_height = 3.0  # maximum height of mixed layer in meters
    
    # 0. find max‐T and first local min depths per profile
    n_prof, n_lev = ct.shape
    depth_max_T = np.full(n_prof, np.nan)
    depth_min_T = np.full(n_prof, np.nan)
    for k in range(n_prof):
        temp = np.ma.masked_invalid(ct[k])
        if temp.count() == 0:
            continue
        idx_max = np.ma.argmax(temp)
        idx_min = np.ma.argmin(temp)
        depth_max_T[k] = p[k, idx_max]
        depth_min_T[k] = p[k, idx_min]
        assert not np.isnan(depth_max_T[k]), f"No maximum-temperature depth for profile {k}"
        assert not np.isnan(depth_min_T[k]), f"No minimum-temperature depth for profile {k}"
    
    # 1) the arrays themselves must exist
    assert depth_min_T is not None and depth_max_T is not None, \
        "depth_min_T or depth_max_T was never initialized"

    # 2) no Python None in any entry
    assert not any(val is None for val in depth_min_T.tolist()), \
        "depth_min_T contains NoneType entries"
    assert not any(val is None for val in depth_max_T.tolist()), \
        "depth_max_T contains NoneType entries"

    # 3) no NaNs slipping through either
    assert not np.isnan(depth_min_T).any(), \
        "depth_min_T contains NaN values"
    assert not np.isnan(depth_max_T).any(), \
        "depth_max_T contains NaN values"

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
    
    #1. Detect mixed layers and interfaces using temperature gradients
    mask_ml, mask_int = detect_mixed_layers_dual(p, ct, smooth, thres_ml, thres_int)

    # 2. fill short gaps in the masks
    mask_ml, mask_int = fill_gaps(mask_ml, mask_int, gap_ml, gap_int)
    mask_ml  = mixed_con(ct, mask_ml, ml_temp)
    mask_int = interface_con(ct, mask_int, int_temp)
    mask_int = interface_height(p, mask_int, ml_height)
    
    assert np.any(mask_ml), "No mixed layer mask found in the data with gradient check."

    # 3. assemble staircase structure
    clean_ml  = np.zeros_like(mask_ml,  dtype=bool)
    clean_int = np.zeros_like(mask_int, dtype=bool)
    mask_cl   = np.zeros_like(mask_ml,  dtype=bool)
    mask_sc   = np.zeros_like(mask_ml,  dtype=bool)

    cl_points = int(np.ceil(cl_length / FIXED_RESOLUTION_METER))

    # 3. prune short runs (thickness condition) :contentReference[oaicite:3]{index=3}
    
    ml_min_grid = int(np.ceil(ml_min_depth / FIXED_RESOLUTION_METER))
    
    for i in range(n_prof):
        arr = np.zeros(n_lev, dtype=int)
        arr[mask_ml[i]]  = 1
        arr[mask_int[i]] = 2

        cleaned = continuity(arr, ml_min_grid, cl_points)
        clean_ml[i]  = (cleaned == 1)
        clean_int[i] = (cleaned == 2)
        mask_cl[i]   = (cleaned == 3)
        mask_sc[i]   = (cleaned > 0)

    masks = SimpleNamespace(ml=clean_ml, int=clean_int,
                                cl=mask_cl, sc=mask_sc)
    return masks, depth_min_T, depth_max_T