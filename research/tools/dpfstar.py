import slam.differential_geometry as sdg
import numpy as np

def compute_dpfstar(mesh, curvature, scaling, list_alpha_ref):
    """
    """
    alphas = [1/np.power(scaling,2) * alpha_ref for alpha_ref in list_alpha_ref]
    # compute dpf
    dpf = sdg.depth_potential_function(mesh, curvature=curvature, alphas=alphas)
    # normalisation
    dpf_star = [-df /scaling for df in dpf]
    return dpf_star
