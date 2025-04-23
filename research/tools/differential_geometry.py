import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg
import trimesh

# Error tolerance for the solver
solver_tolerance = 1e-6

def gaussian_smoothing(mesh, texture, fwhm=10.0):
    """
    Smooth a surface texture using the FEM Laplacian and the heat equation.
    Equivalent to Gaussian smoothing with specified FWHM in mm.

    Args:
        mesh: A trimesh.Trimesh object with .vertices and .faces.
        texture: A numpy array of scalar values (one per vertex).
        fwhm: Full Width Half Maximum in mm for smoothing.

    Returns:
        Smoothed texture (numpy array).
    """
    print("Applying Gaussian smoothing (heat diffusion)...")

    sigma = fwhm / 2.3548  # Convert FWHM to sigma
    dt = sigma ** 2
    nb_iter = 1

    lap, lap_b = compute_mesh_laplacian(mesh, lap_type="fem")
    smoothed = laplacian_smoothing(texture, lap, lap_b, nb_iter, dt)
    return smoothed

def laplacian_smoothing(texture_data, lap, lap_b, nb_iter, dt):
    """
    Perform smoothing using the FEM Laplacian (implicit scheme).
    """
    M = lap_b + dt * lap
    for _ in range(nb_iter):
        b = lap_b @ texture_data
        # Safe fallback without tol argument due to some SciPy versions
        texture_data, _ = cg(M.tocsr(), b)
    return texture_data

def compute_mesh_laplacian(mesh, weights=None, fem_b=None, lap_type="fem"):
    """
    Compute Laplacian of a mesh using FEM or conformal weights.
    """
    print("  Computing Laplacian")
    if weights is None or fem_b is None:
        weights, fem_b = compute_mesh_weights(mesh, weight_type=lap_type)

    if lap_type == "fem":
        weights.data /= 2

    n = weights.shape[0]
    sB = fem_b.sum(axis=0)
    diaB = sparse.dia_matrix((sB, 0), shape=(n, n))
    B = sparse.lil_matrix(diaB + fem_b)
    s = weights.sum(axis=0)
    dia = sparse.dia_matrix((s, 0), shape=(n, n))
    L = sparse.lil_matrix(dia - weights)

    return L, B

def compute_mesh_weights(mesh, weight_type="fem", cot_threshold=None, z_threshold=None):
    """
    Compute sparse weight matrix based on mesh connectivity.
    """
    print(f"    Computing mesh weights of type {weight_type}")
    vert = mesh.vertices
    poly = mesh.faces
    nbv = vert.shape[0]

    W = sparse.lil_matrix((nbv, nbv))
    femB = sparse.lil_matrix((nbv, nbv))

    threshold = 0.0001
    for i in range(3):
        i1 = i
        i2 = (i + 1) % 3
        i3 = (i + 2) % 3
        pp = vert[poly[:, i2], :] - vert[poly[:, i1], :]
        qq = vert[poly[:, i3], :] - vert[poly[:, i1], :]
        cr = np.cross(pp, qq)
        area = np.linalg.norm(cr, axis=1) / 2

        noqq = np.linalg.norm(qq, axis=1)
        nopp = np.linalg.norm(pp, axis=1)
        nopp[nopp < threshold] = threshold
        noqq[noqq < threshold] = threshold

        pp /= nopp[:, np.newaxis]
        qq /= noqq[:, np.newaxis]

        ang = np.arccos(np.sum(pp * qq, axis=1))
        ang[np.isnan(ang)] = threshold
        cot = 1 / np.tan(ang)

        W += sparse.coo_matrix((cot, (poly[:, i2], poly[:, i3])), shape=(nbv, nbv))
        W += sparse.coo_matrix((cot, (poly[:, i3], poly[:, i2])), shape=(nbv, nbv))

        femB += sparse.coo_matrix((area / 12, (poly[:, i2], poly[:, i3])), shape=(nbv, nbv))
        femB += sparse.coo_matrix((area / 12, (poly[:, i3], poly[:, i2])), shape=(nbv, nbv))

    return W.tocsr(), femB.tocsr()