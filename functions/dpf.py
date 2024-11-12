
from scipy.sparse.linalg import lgmres, eigsh
import functions.laplacian as laplacian
solver_tolerance = 1e-6

def depth_potential_function(mesh, curvature, alpha):
    """
    Compute the depth potential function of a mesh as desribed in
    Boucher, M., Whitesides, S., & Evans, A. (2009).
    Depth potential function for folding pattern representation,
    registration and analysis.
    Medical Image Analysis, 13(2), 203–14.
    doi:10.1016/j.media.2008.09.001
    :param mesh: 3D mesh with N vertices
    :param curvature: array (1,N)
    :param alpha: regularisation paramater (0 : concavity, 1: curvature)
    :return: array 
    """
    L, LB = laplacian.compute_mesh_laplacian(mesh, lap_type="fem")
    B = (
        -2
        * LB
        * (curvature - (np.sum(curvature * LB.diagonal()) / np.sum(LB.diagonal())))
    )

    M = alpha * LB + L / 2
    dpf, info = lgmres(M.tocsr(), B, tol=solver_tolerance)
    return dpf