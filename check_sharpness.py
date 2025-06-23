import numpy as np

def check_sharpness(mask_cl, threshold):
    """
    Classify connecting‐layer sequences into “mushy” or “supermushy” based on length.

    Parameters
    ----------
    mask_cl : array_like of bool, shape (n_profiles, n_levels) or (n_levels,)
        Boolean mask marking connection‐layer points.
    threshold : float or int
        Length threshold (in grid‐point counts). Any contiguous run of True
        shorter than (threshold / 2) will be classified as mushy; otherwise
        as supermushy.

    Returns
    -------
    cl_mushy : ndarray of bool, same shape as mask_cl
        True where a connection‐layer run is classified as mushy.
    cl_supermushy : ndarray of bool, same shape as mask_cl
        True where a connection‐layer run is classified as supermushy.
    """
    arr = np.atleast_2d(mask_cl).astype(bool)
    n_prof, n_lev = arr.shape

    cl_mushy      = np.zeros_like(arr, dtype=bool)
    cl_supermushy = np.zeros_like(arr, dtype=bool)
    half_thr = threshold / 2.0

    for i in range(n_prof):
        j = 0
        while j < n_lev:
            if arr[i, j]:
                start = j
                # advance until end of this True‐run
                while j < n_lev and arr[i, j]:
                    j += 1
                end = j  # one past the last True
                length = end - start
                if length < half_thr:
                    cl_mushy[i, start:end] = True
                else:
                    cl_supermushy[i, start:end] = True
            else:
                j += 1

    # if original was 1D, squeeze back
    if mask_cl.ndim == 1:
        return cl_mushy[0], cl_supermushy[0]
    return cl_mushy, cl_supermushy