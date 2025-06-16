import numpy as np

class MixedLayer:
    """
    Holds all mixed‐layer diagnostics for a batch of profiles.
    Attributes are 2D masked arrays of shape (n_profiles, max_layers).
    """
    def __init__(self, n_profiles, max_layers):
        self.T      = np.ma.zeros((n_profiles, max_layers))    # avg temperature
        self.S      = np.ma.zeros((n_profiles, max_layers))    # avg salinity
        self.r      = np.ma.zeros((n_profiles, max_layers))    # avg density
        self.p      = np.ma.zeros((n_profiles, max_layers))    # avg pressure
        self.dT     = np.ma.zeros((n_profiles, max_layers))    # ΔT within layer
        self.dS     = np.ma.zeros((n_profiles, max_layers))    # ΔS within layer
        self.dr     = np.ma.zeros((n_profiles, max_layers))    # Δρ within layer
        self.height = np.ma.zeros((n_profiles, max_layers))    # layer thickness (points)
        self.Tu     = np.ma.zeros((n_profiles, max_layers))    # avg Turner angle
        self.R      = np.ma.zeros((n_profiles, max_layers))    # avg density ratio
        # simple integer counters
        self.count  = np.zeros((n_profiles, max_layers), dtype=int)
        # masks to mark unused entries
        self._mask  = np.ones((n_profiles, max_layers), dtype=bool)

    def mask_unused(self):
        """Apply the internal mask to all arrays."""
        for attr, arr in vars(self).items():
            if isinstance(arr, np.ma.MaskedArray):
                arr.mask = self._mask
        self.count = np.ma.array(self.count, mask=self._mask)

class interface:
    """
    Holds all interface diagnostics.
    Attributes are 2D masked arrays of shape (n_profiles, max_layers).
    """
    def __init__(self, n_profiles, max_layers):
        self.dTdz     = np.ma.zeros((n_profiles, max_layers))
        self.dT       = np.ma.zeros((n_profiles, max_layers))
        self.dS       = np.ma.zeros((n_profiles, max_layers))
        self.dr       = np.ma.zeros((n_profiles, max_layers))
        self.dist     = np.ma.zeros((n_profiles, max_layers))
        self.Tu       = np.ma.zeros((n_profiles, max_layers))
        self.R        = np.ma.zeros((n_profiles, max_layers))
        self.intersect= np.zeros((n_profiles, max_layers), dtype=int)
        self.count    = np.zeros((n_profiles, max_layers), dtype=int)
        self._mask    = np.ones((n_profiles, max_layers), dtype=bool)

    def mask_unused(self):
        for attr, arr in vars(self).items():
            if isinstance(arr, np.ma.MaskedArray):
                arr.mask = self._mask
        self.count = np.ma.array(self.count, mask=self._mask)