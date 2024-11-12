import numpy as np
from scipy import sparse
import scipy.stats.stats as sss
from scipy.sparse.linalg import lgmres, eigsh



solver_tolerance = 1e-6




def compute_mesh_weights(
    mesh, weight_type="conformal", cot_threshold=None, z_threshold=None
):
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

    :param mesh:
    :param weight_type: choice across conformal, fem, meanvalue, authalic
    :param cot_threshold:
    :param z_threshold:
    :return:
    """
    print("    Computing mesh weights of type " + weight_type)
    vert = mesh.vertices
    poly = mesh.faces

    Nbv = vert.shape[0]
    W = sparse.lil_matrix((Nbv, Nbv))
    femB = sparse.lil_matrix((Nbv, Nbv))
    if weight_type == "conformal" or weight_type == "fem":
        threshold = 0.0001  # np.spacing(1)??
        threshold_needed = 0
        for i in range(3):
            i1 = np.mod(i, 3)
            i2 = np.mod(i + 1, 3)
            i3 = np.mod(i + 2, 3)
            pp = vert[poly[:, i2], :] - vert[poly[:, i1], :]
            qq = vert[poly[:, i3], :] - vert[poly[:, i1], :]
            cr = np.cross(pp, qq)
            area = np.sqrt(np.sum(np.power(cr, 2), 1)) / 2
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
            pp = pp / np.vstack((nopp, np.vstack((nopp, nopp)))).transpose()
            qq = qq / np.vstack((noqq, np.vstack((noqq, noqq)))).transpose()
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
            W = W + sparse.coo_matrix(
                (cot, (poly[:, i2], poly[:, i3])), shape=(Nbv, Nbv)
            )
            W = W + sparse.coo_matrix(
                (cot, (poly[:, i3], poly[:, i2])), shape=(Nbv, Nbv)
            )
            femB = femB + sparse.coo_matrix(
                (area / 12, (poly[:, i2], poly[:, i3])), shape=(Nbv, Nbv)
            )
            femB = femB + sparse.coo_matrix(
                (area / 12, (poly[:, i3], poly[:, i2])), shape=(Nbv, Nbv)
            )


        nnz = W.nnz
        if z_threshold is not None:
            z_weights = sss.zscore(W.data)
            inds_out = np.where(np.abs(z_weights) > z_threshold)[0]
            W.data[inds_out] = np.mean(W.data)
            print(
                "    -Zscore threshold needed for ",
                len(inds_out),
                " values = ",
                100 * len(inds_out) / nnz,
                " %",
            )

        print(
            "    -edge length threshold needed for ",
            threshold_needed,
            " values = ",
            100 * threshold_needed / nnz,
            " %",
        )
        if cot_threshold is not None:
            print(
                "    -cot threshold needed for ",
                threshold_needed_angle,
                " values = ",
                100 * threshold_needed_angle / nnz,
                " %",
            )

    if weight_type == "meanvalue":
        for i in range(3):
            i1 = np.mod(i, 3)
            i2 = np.mod(i + 1, 3)
            i3 = np.mod(i + 2, 3)
            pp = vert[poly[:, i2], :] - vert[poly[:, i1], :]
            qq = vert[poly[:, i3], :] - vert[poly[:, i1], :]
            rr = vert[poly[:, i2], :] - vert[poly[:, i3], :]
            # normalize the vectors
            noqq = np.sqrt(np.sum(qq * qq, 1))
            nopp = np.sqrt(np.sum(pp * pp, 1))
            norr = np.sqrt(np.sum(rr * rr, 1))
            pp = pp / np.vstack((nopp, np.vstack((nopp, nopp)))).transpose()
            qq = qq / np.vstack((noqq, np.vstack((noqq, noqq)))).transpose()
            rr = rr / np.vstack((norr, np.vstack((norr, norr)))).transpose()
            # compute angles
            angi1 = np.arccos(np.sum(pp * qq, 1)) / 2
            qq = -qq
            angi2 = np.arccos(np.sum(rr * qq, 1)) / 2
            W = W + sparse.coo_matrix(
                (np.tan(angi1) / norr, (poly[:, i1], poly[:, i3])), shape=(Nbv, Nbv)
            )
            W = W + sparse.coo_matrix(
                (np.tan(angi2) / norr, (poly[:, i3], poly[:, i1])), shape=(Nbv, Nbv)
            )
        nnz = W.nnz
    if weight_type == "authalic":
        for i in range(3):
            i1 = np.mod(i, 3)
            i2 = np.mod(i + 1, 3)
            i3 = np.mod(i + 2, 3)
            pp = vert[poly[:, i2], :] - vert[poly[:, i1], :]
            qq = vert[poly[:, i3], :] - vert[poly[:, i1], :]
            rr = vert[poly[:, i2], :] - vert[poly[:, i3], :]
            # normalize the vectors
            noqq = np.sqrt(np.sum(qq * qq, 1))
            nopp = np.sqrt(np.sum(pp * pp, 1))
            norr = np.sqrt(np.sum(rr * rr, 1))
            pp = pp / np.vstack((nopp, np.vstack((nopp, nopp)))).transpose()
            qq = qq / np.vstack((noqq, np.vstack((noqq, noqq)))).transpose()
            rr = rr / np.vstack((norr, np.vstack((norr, norr)))).transpose()
            # compute angles
            angi1 = np.arccos(np.sum(pp * qq, 1)) / 2
            cot1 = 1 / np.tan(angi1)
            qq = -qq
            angi2 = np.arccos(np.sum(rr * qq, 1)) / 2
            cot2 = 1 / np.tan(angi2)
            W = W + sparse.coo_matrix(
                (cot1 / norr**2, (poly[:, i3], poly[:, i1])), shape=(Nbv, Nbv)
            )
            W = W + sparse.coo_matrix(
                (cot2 / norr**2, (poly[:, i1], poly[:, i3])), shape=(Nbv, Nbv)
            )
        nnz = W.nnz
    li = np.hstack(W.data)
    nb_Nan = len(np.where(np.isnan(li))[0])
    nb_neg = len(np.where(li < 0)[0])
    print(
        "    -number of Nan in weights: ",
        nb_Nan,
        " = ",
        100 *
        nb_Nan /
        nnz,
        " %")
    print(
        "    -number of Negative values in weights: ",
        nb_neg,
        " = ",
        100 * nb_neg / nnz,
        " %",
    )

    return W.tocsr(), femB.tocsr()



def compute_mesh_laplacian(
        mesh, weights=None, fem_b=None, lap_type="conformal"):
    """
    compute laplacian of a mesh
    see compute_mesh_weight for details
    :param mesh: 3D mesh with N vertices
    :param weights:
    :param fem_b:
    :param lap_type:
    :return: arrays L and B
    """
    print("  Computing Laplacian")
    if weights is None:
        (weights, fem_b) = compute_mesh_weights(mesh, weight_type=lap_type)

    if lap_type == "fem":
        weights.data = weights.data / 2

    N = weights.shape[0]
    sB = fem_b.sum(axis=0)
    diaB = sparse.dia_matrix((sB, 0), shape=(N, N))
    B = sparse.lil_matrix(diaB + fem_b)
    s = weights.sum(axis=0)
    dia = sparse.dia_matrix((s, 0), shape=(N, N))
    L = sparse.lil_matrix(dia - weights)

    li = np.hstack(L.data)
    print("    -nb Nan in Laplacian : ", len(np.where(np.isnan(li))[0]))
    print("    -nb Inf in Laplacian : ", len(np.where(np.isinf(li))[0]))

    return L, B