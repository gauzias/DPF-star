import slam.curvature as scurv
import numpy as np
from scipy import sparse
import scipy.stats.stats as ss
from scipy.sparse.linalg import lgmres
import os
import slam.io as sio


########################
# error tolerance for lgmres solver
solver_tolerance = 1e-6
########################

def curv(mesh):
    """
    :param mesh: mesh object
    :return:
    K1 : texture, first principal componant of curvature
    K2 : texture, second principal compopant of curvature
    """
    PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = scurv.curvatures_and_derivatives(mesh)
    K1 = PrincipalCurvatures[0, :]
    K2 = PrincipalCurvatures[1, :]
    return K1,K2


def compute_mesh_weights(mesh, cot_threshold=None, z_threshold=None):
    """
    compute a weight matrix
    W is sparse weight matrix and W(i,j) = 0 is vertex i and vertex j are not
    connected in the mesh.
    details are presented in:
    Desbrun, M., Meyer, M., & Alliez, P. (2002).
    Intrinsic parameterizations of surface meshes.
    Computer Graphics Forum, 21(3), 209–218.
    https://doi.org/10.1111/1467-8659.00580
    and
    Reuter, M., Biasotti, S., & Giorgi, D. (2009).
    Discrete Laplace–Beltrami operators for shape analysis and segmentation.
    Computers & …, 33(3), 381–390.
    https://doi.org/10.1016/j.cag.2009.03.005
    additional checks and thresholds are applied to ensure finite values

    :param mesh: trimesh object
    :param cot_threshold:
    :param z_threshold:
    :return: W, femB
    """

    print("    Computing mesh weights")
    vert = mesh.vertices
    poly = mesh.faces

    Nbv = vert.shape[0]
    W = sparse.lil_matrix((Nbv, Nbv))
    femB = sparse.lil_matrix((Nbv, Nbv))

    threshold = 0.0001  # np.spacing(1)??
    threshold_needed = 0
    for i in range(3):
        # we loop on the three vertices of a face
        # parrallel trough all the faces
        i1 = np.mod(i, 3)
        i2 = np.mod(i + 1, 3)
        i3 = np.mod(i + 2, 3)
        # we consider a triangle face with vtx0, vtx1 and vtx2
        # pp is the vector vtx(i) -> vtx(i+1)
        # qq is the vector vtx(i) -> vtx(i+2)
        pp = vert[poly[:, i2], :] - vert[poly[:, i1], :]
        qq = vert[poly[:, i3], :] - vert[poly[:, i1], :]
        # cross product of pp and qq
        cr = np.cross(pp, qq)
        # area of the triangle : half of the magnitude of the cross product between two adjencent edges
        area = np.sqrt(np.sum(np.power(cr, 2), 1)) / 2
        # norm of qq and pp
        noqq = np.sqrt(np.sum(qq * qq, 1))
        nopp = np.sqrt(np.sum(pp * pp, 1))
        thersh_nopp = np.where(nopp < threshold)[0]
        thersh_noqq = np.where(noqq < threshold)[0]
        if len(thersh_nopp) > 0:
            nopp[thersh_nopp] = threshold
            threshold_needed += len(thersh_nopp)
        if len(thersh_noqq) > 0:
            noqq[thersh_noqq] = threshold
            threshold_needed += len(thersh_noqq)
        # normalisation of pp and qq : now unit vector
        pp = pp / np.vstack((nopp, np.vstack((nopp, nopp)))).transpose()
        qq = qq / np.vstack((noqq, np.vstack((noqq, noqq)))).transpose()
        # angle of the vertex
        ang = np.arccos(np.sum(pp * qq, 1))
        # ############## preventing infs in weights
        inds_zeros = np.where(ang == 0)[0]
        ang[inds_zeros] = threshold
        threshold_needed_angle = len(inds_zeros)
        ################################
        cot = 1 / np.tan(ang)
        if cot_threshold is not None:
            thresh_inds = cot < 0
            cot[thresh_inds] = cot_threshold
            threshold_needed_angle += np.count_nonzero(thresh_inds)
        W = W + sparse.coo_matrix((cot, (poly[:, i2], poly[:, i3])), shape=(Nbv, Nbv))
        W = W + sparse.coo_matrix((cot, (poly[:, i3], poly[:, i2])), shape=(Nbv, Nbv))
        femB = femB + sparse.coo_matrix((area / 12, (poly[:, i2], poly[:, i3])), shape=(Nbv, Nbv))
        femB = femB + sparse.coo_matrix((area / 12, (poly[:, i3], poly[:, i2])), shape=(Nbv, Nbv))

    nnz = W.nnz
    if z_threshold is not None:
        z_weights = ss.zscore(W.data)
        inds_out = np.where(np.abs(z_weights) > z_threshold)[0]
        W.data[inds_out] = np.mean(W.data)
        print(" -Zscore threshold needed for ",len(inds_out)," values = ",100 * len(inds_out) / nnz," %")
    print(" -edge length threshold needed for ",threshold_needed," values = ",100 * threshold_needed / nnz," %")
    if cot_threshold is not None:
        print(" -cot threshold needed for ",threshold_needed_angle," values = ",100 * threshold_needed_angle / nnz," %")

    li = np.hstack(W.data)
    nb_Nan = len(np.where(np.isnan(li))[0])
    nb_neg = len(np.where(li < 0)[0])
    print("    -number of Nan in weights: ",nb_Nan," = ",100 *nb_Nan/nnz," %")
    print("    -number of Negative values in weights: ",nb_neg," = ",100 * nb_neg / nnz," %")

    return W.tocsr(), femB.tocsr()

def compute_mesh_laplacian( mesh, weights=None, fem_b=None):
    """
    compute laplacian of a mesh
    see compute_mesh_weight for details
    :param mesh: trimesh object
    :param weights: W, femB
    :return: L, B
    """
    print("  Computing Laplacian")
    if weights is None:
        (weights, fem_b) = compute_mesh_weights(mesh)

    N = weights.shape[0]
    sB = fem_b.sum(axis=0)
    # dia_matrix(data,offset) offset=0 is for put the diagonal on the diagonal ^^'
    diaB = sparse.dia_matrix((sB, 0), shape=(N, N))
    # femB has 0 on its diagonal
    B = sparse.lil_matrix(diaB + fem_b)
    s = weights.sum(axis=0)
    dia = sparse.dia_matrix((s, 0), shape=(N, N))
    L = sparse.lil_matrix(dia - weights)

    li = np.hstack(L.data)
    print("    -nb Nan in Laplacian : ", len(np.where(np.isnan(li))[0]))
    print("    -nb Inf in Laplacian : ", len(np.where(np.isinf(li))[0]))

    return L, B


def depth_potential_function(mesh, curvature, alphas=[0.03]):
    """
    compute the depth potential function of a mesh as desribed in
    Boucher, M., Whitesides, S., & Evans, A. (2009).
    Depth potential function for folding pattern representation,
    registration and analysis.
    Medical Image Analysis, 13(2), 203–14.
    doi:10.1016/j.media.2008.09.001
    :param mesh:
    :param curvature:
    :param alphas:
    :return:
    """
    L, LB = compute_mesh_laplacian(mesh)
    B = (
        -2
        * LB
        * (curvature - (np.sum(curvature * LB.diagonal()) / np.sum(LB.diagonal())))
    )
    # be careful with factor 2 used in eq (13)

    dpf = []
    for ind, alpha in enumerate(alphas):
        M = alpha * LB + L / 2
        dpf_t, info = lgmres(M.tocsr(), B, tol=solver_tolerance)
        dpf.append(dpf_t)
    return dpf


def dpfstar(mesh, source, alphas=[500], adaptation = 'volume_hull'):
    """
    compute the depth potential function of a mesh. The scale of interest is adapted to
    the size of the mesh.
    :param mesh: TRIMESH MESH
    :param curvature: TEXTURE (darray)
    :param alphas: LIST : adapt the size of interest for computing sulcal depth
    :return: TEXTURE (darray)
    """
    hull = mesh.convex_hull
    vol_hull = hull.volume
    vol = mesh.volume
    surface = mesh.area
    # adaptation of the scale of interest
    if adaptation == 'volume_hull':
        lc = np.power(vol_hull, 1/3)
    if adaptation =='volume':
        lc = np.power(vol,1/3 )
    if adaptation =='surface':
        lc = np.power(surface, 1/2)

    # compute the laplacian
    L, LB = compute_mesh_laplacian(mesh)

    # dedimensialisation of the laplacian and the curvature
    L = L * np.square(lc)
    source = source * lc

    # compute the dpf
    B = ( LB * ( source -( np.sum(source * LB.diagonal()) / np.sum(LB.diagonal()) ) ) )
    # be careful with factor 2 used in eq (13)

    dpf = []
    for ind, alpha in enumerate(alphas):
        M = (alpha * LB) + L
        dpf_t, info = lgmres(M.tocsr(), B, tol=solver_tolerance)
        dpf.append(dpf_t)

    return dpf



