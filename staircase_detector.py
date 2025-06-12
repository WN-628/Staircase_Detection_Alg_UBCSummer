import numpy as np

def cent_derivative2d(f,z):
    dz1     = z[:,1:-1]-z[:,:-2]
    dz2     = z[:,2:]-z[:,1:-1]
    fkp1    = f[:,2:]
    fkm1    = f[:,:-2]
    fk      = f[:,1:-1]

    # dfdz    = np.ma.zeros(f.shape)
    dfdz = np.ma.array(np.zeros(f.shape), mask=np.zeros(f.shape, dtype=bool))
    dfdz[:,1:-1] = ( dz1**2*fkp1 + ( dz2**2-dz1**2 )*fk - dz2**2*fkm1) / ( dz1*dz2*(dz1+dz2) )
    dfdz.mask[dfdz==0]=True 
    return dfdz

def get_mixed_layers_simple(p, ct, c1, c2, c3):
    """
    p   : 2D array, shape (Nobs, Nlev) -> depth (metres or dbar) increasing downward
    ct  : 2D array, same shape -> conservative temperature

    c1  : threshold for |dT/dz| defining mixed-layer points
    c2  : threshold for |dT/dz| defining gradient-layer points
    c3  : max vertical separation (same units as p) to link gradient->mixed

    Returns:
        gl_inds     : list of arrays of gradient-layer indices
        ml_inds     : list of arrays of mixed-layer indices
        masks       : namespace with boolean arrays .ml, .gl, .cl, .sc each shape (Nobs, Nlev)
        t_max_depth : 1D array of depths of maximum temperature per profile
        t_min_depth : 1D array of depths of first local minimum above max per profile
    """
    Nobs, Nlev = ct.shape

    # 1) locate max and first local min temps above it (shallower depths)
    # to store the max and min temperature for temperature profile
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
    # sa = np.ma.masked_where(~valid_depth_mask, sa)

    # 2) compute gradient via cent_derivative2d
    dTdz = cent_derivative2d(ct, p)

    # 3) initialize masks
    mask_ml = np.abs(dTdz) < c1
    mask_gl = np.abs(dTdz) > c2
    mask_cl = np.zeros_like(mask_ml, dtype=bool)
    mask_sc = np.zeros_like(mask_ml, dtype=bool)

    ml_inds = []
    gl_inds = []

    # 4) per-profile QC, linkage, and staircase check
    for i in range(Nobs):
        # define region: [min_depth-25, max_depth+25]

        ml = np.where(mask_ml[i])[0]
        gl = np.where(mask_gl[i])[0]
        ml_inds.append(ml)
        gl_inds.append(gl)

        if ml.size == 0 or gl.size == 0:
            continue

        # QC-filter: gradient must exceed local ML gradient and be within c3
        local_max_ml = np.max(np.abs(dTdz[i, ml]))
        passed = [idx for idx in gl
                    if np.abs(dTdz[i, idx]) > local_max_ml
                        and np.min(np.abs(p[i, ml] - p[i, idx])) <= c3]

        if len(passed) < 2:
            continue

        # build alternating primary sequence
        prim = np.sort(np.concatenate((ml, passed)))
        seq_count = 0
        last_type = None
        seq_locs = []
        for j in prim:
            curr_type = 'ml' if j in ml else 'gl'
            if curr_type != last_type:
                seq_count += 1
                seq_locs.append(j)
                last_type = curr_type
            else:
                break

        # mark valid staircase
        if seq_count >= 3:
            mask_sc[i, seq_locs] = True
            mask_cl[i, passed] = True

    # 5) pack masks
    class Masks: pass
    masks = Masks()
    masks.ml = mask_ml
    masks.gl = mask_gl
    masks.cl = mask_cl
    masks.sc = mask_sc
    
    
    max_idx = np.ma.argmax(temp_raw)
    min_idx = np.ma.argmin(temp_raw)

    depth_max_T[k] = pressure_raw[max_idx]
    depth_min_T[k] = pressure_raw[min_idx]

    return gl_inds, ml_inds, masks, depth_min_T, depth_max_T

