import numpy as np
import scipy.ndimage as ndimage
from types import SimpleNamespace

def mask_continuity(mask, length):
    '''
    For each row in the 2D boolean array `mask`, find all contiguous runs of True.
    If a run’s length is < `length`, set those positions to False.
    
    Args:
        mask:   2D boolean array of shape (n_profiles, n_levels).
        length: minimum run‐length to keep.
        
    Returns:
        A new 2D boolean array of the same shape, with short runs removed.
    '''
    # ensure a plain boolean ndarray
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
                while j < n_cols and row[j]:
                    j += 1
                run_len = j - start
                if run_len < length:
                    out[i, start:j] = False
            else:
                j += 1
    
    return out

def mask_sc_continuity(mask_ml, mask_gl, num_ml, num_gl, num_cl=3):
    '''
    Check for valiud staircase structures based on given mixed-layer (ml) and gradient-layer (gl) masks. 
    
    Args:
        mask_ml (2D bool): mixed-layer mask
        mask_gl (2D bool): gradient-layer mask
        num_cl (int): maximum allowed gap (number of grid points) between layers
    
    Conditions:
        - ml and gl runs must be at least 3 grid points long
        - ml and gl runs can be connected by a gap of at most `num_cl` grid points
        - a valid staircase must have at least 3 alternating runs of ml/gl
    
    Returns:
        mask_cl (2D bool): True at connection layer positions
        clean_ml (2D bool): ml mask pruned for valid staircases
        clean_gl (2D bool): gl mask pruned for valid staircases
        mask_sc (2D bool): combined staircase mask (True where sc > 0)
    '''
    
    # Ensure that mask_ml and mask_gl are has valid lengths 
    mask_ml = mask_continuity(mask_ml, num_ml)
    mask_gl = mask_continuity(mask_gl, num_gl)
    
    # Convert masks to boolean arrays
    ml = np.asarray(mask_ml, dtype=bool)
    gl = np.asarray(mask_gl, dtype=bool)
    n_rows, n_cols = ml.shape

    # Create an 2D integer map: 1=ml, 2=gl, 0=none based on the input mask_ml and mask_gl
    sc = np.zeros((n_rows, n_cols), dtype=int)
    sc[ml] = 1
    sc[gl] = 2

    clean_ml = np.zeros_like(ml, dtype=bool)
    clean_gl = np.zeros_like(gl, dtype=bool)

    def find_runs_with_type(arr):
        '''
        Find contiguous runs (segment) of non-zero values in a 1D array.
        Returns a list of tuples (start_index, end_index, value) for each run.
        '''
        runs = []
        i = 0
        while i < len(arr):
            if arr[i] != 0:
                val = arr[i]
                start = i
                while i < len(arr) and arr[i] == val:
                    i += 1
                runs.append((start, i-1, val))
            else:
                i += 1
        return runs

    for i in range(n_rows):
        runs = find_runs_with_type(sc[i])
        for start_idx in range(len(runs)):
            chain = [runs[start_idx]]
            for next_idx in range(start_idx+1, len(runs)):
                ps, pe, pt = chain[-1]
                cs, ce, ct = runs[next_idx]
                gap = cs - pe - 1
                if ct != pt and gap <= num_cl:
                    chain.append(runs[next_idx])
                else:
                    break
            if len(chain) >= 3:
                for s, e, t in chain:
                    if t == 1:
                        clean_ml[i, s:e+1] = True
                    else:
                        clean_gl[i, s:e+1] = True
                for idx in range(len(chain)-1):
                    pe = chain[idx][1]
                    cs = chain[idx+1][0]
                    sc[i, pe+1:cs] = 3  # mark connection layers as 3

    # Create the final masks
    mask_cl = (sc == 3)
    mask_sc  = clean_ml | clean_gl | mask_cl
    
    return clean_ml, clean_gl, mask_cl, mask_sc

def get_number_layers(num, thres_ml = 10, thres_gl = 20, num_cl = 3):
    class masks: pass
    
    mask_ml = np.abs(num) < thres_ml
    mask_gl = np.abs(num) > thres_gl
    
    # clean the masks for mixed and gradient layers by required thickness
    ml_final, gl_final, cl_final, mask_sc = mask_sc_continuity(mask_ml, mask_gl, 5, 5, num_cl)
    
    ml_final = mask_ml
    gl_final = mask_gl
    
    masks = SimpleNamespace(ml=ml_final,
                            gl=gl_final,
                            cl=cl_final,
                            sc=mask_sc)

    return masks

# 1. Generate a randomized 2D float array
profile = np.array([5]*4 + [25]*6 + [12.23514]*2 + [-4]*5 + [11.5123412351251] * 1 + [-21] + [-29] + [26] * 3, dtype=float)
profile_2d = profile[np.newaxis, :] 

# 2. Define thresholds and continuity
thres_ml = 6     # mark ml where abs(value) < 5
thres_gl = 20    # mark gl where abs(value) > 10
num_cl   = 3     # maximum gap between layers to consider them connected

# 3. Pass through get_number_layers
masks = get_number_layers(profile_2d, thres_ml=thres_ml, thres_gl=thres_gl, num_cl=num_cl)

# 4. Print out the data and resulting masks
print("PROFILE:     ", profile.astype(int))
print("ML MASK:     ", masks.ml.astype(int))
print("GL MASK:     ", masks.gl.astype(int))
print("CL MASK:     ", masks.cl.astype(int))
print("SC MASK:     ", masks.sc.astype(int))