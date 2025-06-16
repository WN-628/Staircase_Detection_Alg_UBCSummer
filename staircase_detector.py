import numpy as np
# import scipy.ndimage as ndimage
from itertools import combinations
from types import SimpleNamespace

def cent_derivative2d(f,z):
    # check if f and z are 2D arrays with the same shape
    if f.shape != z.shape:
        raise ValueError(f"f and z must have the same shape; got f{f.shape}, z{z.shape}")
    if not (np.issubdtype(f.dtype, np.floating) and np.issubdtype(z.dtype, np.floating)):
        raise TypeError("Both f and z must be float arrays (float32 or float64).")

    dfdz = np.zeros_like(f)  # automatically float, since f is float
    
    dz1     = z[:,1:-1]-z[:,:-2]
    dz2     = z[:,2:]-z[:,1:-1]
    fkp1    = f[:,2:]
    fkm1    = f[:,:-2]
    fk      = f[:,1:-1]

    # dfdz    = np.ma.zeros(f.shape)
    # dfdz = np.ma.array(np.zeros(f.shape), np.nan)
    dfdz[:,1:-1] = ( dz1**2*fkp1 + ( dz2**2-dz1**2 )*fk - dz2**2*fkm1) / ( dz1*dz2*(dz1+dz2) )
    # dfdz.mask[:, [0, -1]] = True # Mask out the boundaries
    # dfdz.mask[dfdz==0]=True 
    
    # Boundary handling: forward/backward difference
    dfdz[:, 0] = (f[:, 1] - f[:, 0]) / (z[:, 1] - z[:, 0])
    dfdz[:, -1] = (f[:, -1] - f[:, -2]) / (z[:, -1] - z[:, -2])
    
    print(f"🧮 Gradient min: {np.nanmin(dfdz)}, max: {np.nanmax(dfdz)}")
    
    return dfdz

# Define the fast all‐chains function
def remove_short_runs(mask, min_len):
    mask = np.asarray(mask, bool)
    out = np.zeros_like(mask)
    nR, nC = mask.shape
    for i in range(nR):
        j = 0
        while j < nC:
            if mask[i, j]:
                s = j
                while j < nC and mask[i, j]:
                    j += 1
                e = j
                if e - s >= min_len:
                    out[i, s:e] = True
            else:
                j += 1
    return out

def find_runs(row):
    runs = []
    j = 0
    L = len(row)
    while j < L:
        if row[j] != 0:
            t = row[j]; s = j
            while j < L and row[j] == t:
                j += 1
            runs.append((s, j, t))
        else:
            j += 1
    return runs

def mask_sc_continuity(mask_ml, mask_gl, num_ml, num_gl, num_cl=3):
    ml = remove_short_runs(mask_ml, num_ml)
    gl = remove_short_runs(mask_gl, num_gl)
    nR, nC = ml.shape

    sc_map = np.zeros((nR, nC), int)
    sc_map[ml] = 1
    sc_map[gl] = 2

    clean_ml = np.zeros_like(ml)
    clean_gl = np.zeros_like(gl)
    mask_cl  = np.zeros_like(ml)

    for i in range(nR):
        while True:
            runs = find_runs(sc_map[i])
            R = len(runs)
            if R < 3:
                break

            # DP on runs
            dp   = [1]*R
            prev = [-1]*R
            for j in range(R):
                sj, ej, tj = runs[j]
                for k in range(j):
                    sk, ek, tk = runs[k]
                    if tk != tj and sj - ek <= num_cl:
                        if dp[k] + 1 > dp[j]:
                            dp[j] = dp[k] + 1
                            prev[j] = k

            # find maximal chain
            j_max = max(range(R), key=lambda x: dp[x])
            if dp[j_max] < 3:
                break

            # backtrack full chain
            chain = []
            cur = j_max
            while cur >= 0:
                chain.append(cur)
                cur = prev[cur]
            chain.reverse()

            # mark all sub-chains >=3
            K = len(chain)
            for start in range(K-2):
                for end in range(start+3, K+1):
                    sub = chain[start:end]
                    for idx in sub:
                        s, e, t = runs[idx]
                        if t == 1:
                            clean_ml[i, s:e] = True
                        else:
                            clean_gl[i, s:e] = True
                    for a, b in zip(sub, sub[1:]):
                        _, e0, _ = runs[a]
                        s1, _, _ = runs[b]
                        if s1 > e0:
                            mask_cl[i, e0:s1] = True

            # remove entire chain and gaps
            for idx in chain:
                s, e, _ = runs[idx]
                sc_map[i, s:e] = 0
            for a, b in zip(chain, chain[1:]):
                _, e0, _ = runs[a]
                s1, _, _ = runs[b]
                if s1 > e0:
                    sc_map[i, e0:s1] = 0

    mask_sc = clean_ml | clean_gl | mask_cl
    return clean_ml, clean_gl, mask_cl, mask_sc

def get_mixed_layers(p, ct, thres_ml=0.0002, thres_gl=0.005, num_cl=3):
    '''
    p   : 2D array, shape (Nobs, Nlev) -> depth (metres or dbar) increasing downward
    ct  : 2D array, same shape -> conservative temperature

    thres_ml  : threshold for |dT/dz| defining mixed-layer points
    thres_gl  : threshold for |dT/dz| defining gradient-layer points
    depth_cl  : max vertical separation (same units as p) to link gradient->mixed

    Returns:
        gl_inds     : list of arrays of gradient-layer indices
        ml_inds     : list of arrays of mixed-layer indices
        masks       : namespace with boolean arrays .ml, .gl, .cl, .sc each shape (Nobs, Nlev)
        t_max_depth : 1D array of depths of maximum temperature per profile
        t_min_depth : 1D array of depths of first local minimum above max per profile
    '''
    
    ct_orig = np.copy(ct)
    p_orig = np.copy(p)
    
    '''
    #0 locate max and first local min temps above it (shallower depths)
    to store the max and min temperature for temperature profile
    We only try to find staircase in the region 25m above the max and below the min temperature in depth
    '''
    depth_max_T = np.full(ct.shape[0], np.nan)
    depth_min_T = np.full(ct.shape[0], np.nan)

    for k in range(ct.shape[0]):
        temp_raw = ct[k, :]
        pressure_raw = p[k, :]
        temp_raw = np.ma.masked_invalid(temp_raw)
        pressure_raw = np.ma.masked_where(temp_raw.mask, pressure_raw)

        if temp_raw.count() == 0:
            continue

        max_idx = np.ma.argmax(temp_raw)
        min_idx = np.ma.argmin(temp_raw)
        depth_max_T[k] = pressure_raw[max_idx]
        depth_min_T[k] = pressure_raw[min_idx]

    valid_depth_mask = np.zeros_like(p, dtype=bool)
    for k in range(ct.shape[0]):
        dmin = depth_min_T[k] + 25
        dmax = depth_max_T[k] - 25
        valid_depth_mask[k, :] = (p[k, :] >= dmin) & (p[k, :] <= dmax)

    ct = np.ma.masked_where(~valid_depth_mask, ct)
    p = np.ma.masked_where(~valid_depth_mask, p)
    
    # Ensure that the updated ct and p are still arrays of the same shape
    if ct.shape != p.shape:
        raise ValueError(f"ct and p must have the same shape after masking; got ct{ct.shape}, p{p.shape}")
    
    '''
    step 0A: define classes
    '''
    
    class ml: pass
    class gl: pass
    class masks: pass
    
    '''
    step 1: Mixed layer (ml) and gradient layer (gl) detection
    '''
    # 1a. compute vertical gradients
    dTdz = cent_derivative2d(ct, p)
    
    mask_ml = np.abs(dTdz) < thres_ml
    assert np.any(mask_ml), "No mixed layer mask found in the data based on the gradient condition."
    mask_gl = np.abs(dTdz) > thres_gl
    assert np.any(mask_gl), "No gradient layer mask found in the data based on the gradient condition."
    
    num_ml = 3  # minimum length of one mixed layer in grid points
    num_gl = 3  # minimum length of one gradient layer in grid points
    
    # clean the masks for mixed and gradient layers by required thickness
    ml_final, gl_final, cl_final, mask_sc = mask_sc_continuity(mask_ml, mask_gl, num_ml, num_gl, num_cl)
    
    assert np.any(ml_final), "No mixed layer mask found in the data."
    assert np.any(gl_final), "No gradient layer mask found in the data."
    assert np.any(cl_final), "No connection layer mask found in the data."

    masks = SimpleNamespace(ml=ml_final,
                            gl=gl_final,
                            cl=cl_final,
                            sc=mask_sc)

    return masks, depth_min_T, depth_max_T