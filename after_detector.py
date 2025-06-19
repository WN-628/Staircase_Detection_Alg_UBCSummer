import numpy as np

from staircase_detector import detect_layer_lengths

def count(mask):
    """
    Count continuous True‐runs in a boolean mask.
    
    Parameters
    ----------
    mask : array_like of bool
        1D or 2D boolean array where True marks a layer point.
    
    Returns
    -------
    counts : ndarray, shape (n_profiles,)
        Number of runs of True in each row.
    starts : ndarray, shape (n_profiles, max_runs)
        Start indices of each run, padded with -1.
    ends : ndarray,   shape (n_profiles, max_runs)
        End   indices of each run, padded with -1.
    """
    m = np.asarray(mask, dtype=bool)
    
    # ensure 2D
    if m.ndim == 1:
        m = m[np.newaxis, :]
    elif m.ndim != 2:
        raise ValueError("mask must be 1D or 2D boolean array")
    
    n_profiles, n_levels = m.shape
    counts = np.zeros(n_profiles, dtype=int)
    starts_list, ends_list = [], []
    
    for i in range(n_profiles):
        row = m[i]
        # pad with False at both ends
        padded = np.r_[False, row, False]
        diff   = np.diff(padded.astype(int))
        
        # +1 where False→True (run start), -1 where True→False (run end+1)
        run_starts = np.where(diff ==  1)[0]
        run_ends   = np.where(diff == -1)[0] - 1
        
        counts[i] = len(run_starts)
        starts_list.append(run_starts)
        ends_list.append(run_ends)
    
    max_runs = counts.max()
    starts = np.full((n_profiles, max_runs), -1, dtype=int)
    ends   = np.full((n_profiles, max_runs), -1, dtype=int)
    
    for i in range(n_profiles):
        k = counts[i]
        starts[i, :k] = starts_list[i]
        ends[i,   :k] = ends_list[i]
    
    return counts, starts, ends

def extract_length(mask, p):
    """
    Extract the physical thicknesses of each contiguous True-run in a boolean mask,
    by calling detect_layer_lengths.

    Parameters
    ----------
    mask : ndarray of bool, shape (n_profiles, n_levels) or (n_levels,)
        Boolean mask array where True indicates the presence of a layer.
    p : ndarray of float, same shape as mask or shape (n_levels,)
        Depth or pressure values corresponding to each level; must be increasing downward.

    Returns
    -------
    lengths : list or list of lists
        - If `mask` is 1D, returns a list of floats: the thickness of each run.
        - If `mask` is 2D (n_profiles × n_levels), returns a list of length n_profiles:
            each element is a list of thicknesses for that profile’s runs.
    """
    runs = detect_layer_lengths(mask, p)
    # single-profile case
    if isinstance(runs, dict):
        return runs['thickness'].tolist()
    # multi-profile case
    return [prof_run['thickness'].tolist() for prof_run in runs]

def extract_temp_width(mask, ct):
    """
    Extract the temperature width of each contiguous True-run in a boolean mask,
    by calling detect_layer_lengths with ct as the “p” array.

    Parameters
    ----------
    mask : ndarray of bool, shape (n_profiles, n_levels) or (n_levels,)
        Boolean mask array where True indicates the presence of a layer.
    ct : ndarray of float, same shape as mask or shape (n_levels,)
        Conservative temperature values corresponding to each level.

    Returns
    -------
    widths : list or list of lists
        - If `mask` is 1D, returns a list of floats: the temperature width of each run.
        - If `mask` is 2D (n_profiles × n_levels), returns a list of length n_profiles;
            each element is a list of temperature widths for that profile’s runs.
    """
    runs = detect_layer_lengths(mask, ct)
    # single-profile case
    if isinstance(runs, dict):
        return runs['thickness'].tolist()
    # multi-profile case
    return [prof_run['thickness'].tolist() for prof_run in runs]