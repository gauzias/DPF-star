import numpy as np
from scipy import sparse
import scipy.stats.stats as sss
from scipy.sparse.linalg import lgmres, eigsh
import trimesh
import slam.topology as stop
import scipy
import scipy.stats as ss

########################
# error tolerance for lgmres solver
solver_tolerance = 1e-6
########################


def mesh_laplacian_eigenvectors(mesh, nb_vectors=1):
    """
    compute the nb_vectors first non-null eigenvectors of the graph Laplacian
     of mesh
    :param mesh:
    :param nb_vectors:
    :return:
    """
    lap, lap_b = compute_mesh_laplacian(mesh, lap_type="fem")
    w, v = eigsh(lap.tocsr(), nb_vectors +
                 1, M=lap_b.tocsr(), sigma=solver_tolerance)
    return v[:, 1:]



def laplacian_mesh_smoothing(mesh, nb_iter, dt, volume_preservation=False):
    """
    smoothing the mesh by solving the heat equation using fem Laplacian
    ADD REF
    :param mesh:
    :param nb_iter:
    :param dt:
    :return:
    """
    print("    Smoothing mesh")
    lap, lap_b = compute_mesh_laplacian(mesh, lap_type="fem")
    smoothed_vert = laplacian_smoothing(mesh.vertices, lap, lap_b, nb_iter, dt)
    if volume_preservation:
        vol_ini = mesh.volume
        vol_new = trimesh.triangles.mass_properties(
            smoothed_vert[mesh.faces], skip_inertia=True
        )["volume"]
        # scale by volume ratio
        smoothed_vert *= (vol_ini / vol_new) ** (1.0 / 3.0)
    return trimesh.Trimesh(
        faces=mesh.faces, vertices=smoothed_vert, metadata=mesh.metadata, process=False
    )


def laplacian_texture_smoothing(mesh, tex, nb_iter, dt):
    """
    smoothing the texture by solving the heat equation using fem Laplacian
    :param mesh:
    :param tex:
    :param nb_iter:
    :param dt:
    :return:
    """
    print("    Smoothing texture")
    lap, lap_b = compute_mesh_laplacian(mesh, lap_type="fem")
    return laplacian_smoothing(tex, lap, lap_b, nb_iter, dt)


def laplacian_smoothing(texture_data, lap, lap_b, nb_iter, dt):
    """
    sub-function for smoothing using fem Laplacian
    :param texture_data:
    :param lap:
    :param lap_b:
    :param nb_iter:
    :param dt:
    :return:
    """
    mod = 1
    if nb_iter > 10:
        mod = 10
    if nb_iter > 100:
        mod = 100
    if nb_iter > 1000:
        mod = 1000

    M = lap_b + dt * lap
    for i in range(nb_iter):
        texture_data = lap_b * texture_data
        if texture_data.ndim > 1:
            for d in range(texture_data.shape[1]):
                texture_data[:, d], infos = lgmres(
                    M.tocsr(), texture_data[:, d], tol=solver_tolerance
                )
        else:
            texture_data, infos = lgmres(
                M.tocsr(), texture_data, tol=solver_tolerance)
        if i % mod == 0:
            print(i)

    print("    OK")
    return texture_data

"""
def compute_mesh_weights_gradient(mesh, weight_type="conformal", cot_threshold=None, z_threshold=None):
  
    #compute gradient operator
    #W is sparse weight matrix and W(i,j) = 0 is vertex i and vertex j are not
    #connected in the mesh.
    #details are presented in:
    #Discrete Differential Operators on Polygonal Meshes
    #FERNANDO DE GOES, Pixar Animation Studios
    #ANDREW BUTTS, Pixar Animation Studios
    #MATHIEU DESBRUN, ShanghaiTech/Caltech
  

    print("    Computing gradient operator")
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
        # vpp is the coordinate vector vtx(1) -> vtx(2)
        # vqq is the coordiate vector vtx(2) -> vtx(3)
        # vrr is the coordiate vector vtx(3) -> vtx(1)
        Epp = vert[poly[:, i2], :] - vert[poly[:, i1], :]
        Eqq = vert[poly[:, i3], :] - vert[poly[:, i2], :]
        Err = vert[poly[:, i1], :] - vert[poly[:, i3], :]
        # Mean vertex
        App = (vert[poly[:, i2], :] + vert[poly[:, i1], :])/2
        Aqq = (vert[poly[:, i3], :] + vert[poly[:, i2], :]) / 2
        Arr = (vert[poly[:, i1], :] + vert[poly[:, i3], :]) / 2
        # cross product of vpp and vrr
        cr = np.cross(Epp, -Err)
        # area of the triangle : half of the magnitude of the cross product between two adjencent edges
        area = np.sqrt(np.sum(np.power(cr, 2), 1)) / 2
        # gradient
        grad = Epp
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
            (cot, (poly[:, i2], poly[:, i3])), shape=(Nbv, Nbv))
        W = W + sparse.coo_matrix(
            (cot, (poly[:, i3], poly[:, i2])), shape=(Nbv, Nbv))
        femB = femB + sparse.coo_matrix((area / 12, (poly[:, i2], poly[:, i3])), shape=(Nbv, Nbv))
        femB = femB + sparse.coo_matrix((area / 12, (poly[:, i3], poly[:, i2])), shape=(Nbv, Nbv))
    return W.tocsr(), femB.tocsr()
"""

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
    :param mesh:
    :param weights:
    :param fem_b:
    :param lap_type:
    :return:
    """
    print("  Computing Laplacian")
    if weights is None:
        (weights, fem_b) = compute_mesh_weights(mesh, weight_type=lap_type)

    if lap_type == "fem":
        weights.data = weights.data / 2

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


def depth_potential_function(mesh, curvature, alphas):
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
    L, LB = compute_mesh_laplacian(mesh, lap_type="fem")
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

def depth_potential_function_filtre_passe_bande(mesh, curvature, alphas, betas):
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

    hull = mesh.convex_hull
    vol = hull.volume
    lc = np.power(vol, 1/3)
    #area = mesh.area
    #lc = np.sqrt(area)

    L, LB = compute_mesh_laplacian(mesh, lap_type="fem")
    L = L * np.square(lc)
    curvature = curvature * lc

    ## passe haut
    dpf = []
    for inda, alpha in enumerate(alphas):
        for indb, beta in enumerate(betas):
            B1 = LB * L * curvature
            M1 = LB * (alpha*LB + L)
            dpf1, info = lgmres(M1.tocsr(), B1, tol=solver_tolerance)
            ## pass bas
            B2 = LB *  dpf1
            M2 = (beta * LB) + L
            dpf2, info = lgmres(M2.tocsr(), B2, tol=solver_tolerance)
            dpf2 = scipy.stats.zscore(dpf2)
            dpf.append(dpf2)
    return dpf


def depth_potential_function_filtre_passe_haut(mesh, curvature, alphas):
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

    hull = mesh.convex_hull
    vol = hull.volume
    lc = np.power(vol, 1/3)
    #area = mesh.area
    #lc = np.sqrt(area)

    L, LB = compute_mesh_laplacian(mesh, lap_type="fem")

    L = L * np.square(lc)
    curvature = curvature * lc
    B = LB * L * curvature
    # be careful with factor 2 used in eq (13)

    dpf = []
    for ind, alpha in enumerate(alphas):
        M = LB * (alpha*LB + L)
        dpf_t, info = lgmres(M.tocsr(), B, tol=solver_tolerance)
        dpf.append(dpf_t)
    return dpf


def depth_potential_function_anisotropic(mesh, curvature, alphas):
    """
    anisotropic depth potential function
    """

    L, LB = compute_mesh_laplacian(mesh, lap_type="fem")

    B = (
        LB
        * (
                curvature -
                ( np.sum(curvature * LB.diagonal()) / np.sum(LB.diagonal()) )
        )
    )
    # be careful with factor 2 used in eq (13)

    N = LB.shape[0]

    #K = np.abs(curvature)
    K = curvature
    K[K<0] = 0
    K = np.abs(curvature)
    Kmax = np.percentile(K,95)
    Kmin = np.min(K)
    Kone = ((K - Kmin) / (Kmax - Kmin))

    b = 0
    dpf=[]
    for ind, a in enumerate(alphas):

        alpha_diag = -(a - b) * Kone + a
        alpha_diag[alpha_diag < 0] = 0
        print(alpha_diag)
        alpha_matrix = sparse.dia_matrix((alpha_diag, 0), shape=(N, N))
        M = alpha_matrix.multiply(LB) + L
        dpf_t, info = lgmres(M.tocsr(), B, tol=solver_tolerance)
        dpf_t = ss.zscore(dpf_t)
        dpf.append(dpf_t)

    return dpf

def depth_potential_function_normalised(mesh, curvature, alphas):
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

    hull = mesh.convex_hull
    vol = hull.volume
    lc = np.power(vol, 1/3)
    #area = mesh.area
    #lc = np.sqrt(area)

    L, LB = compute_mesh_laplacian(mesh, lap_type="fem")

    L = L * np.square(lc)
    curvature = curvature * lc
    B = (
        LB
        * (
                curvature -
                ( np.sum(curvature * LB.diagonal()) / np.sum(LB.diagonal()) )
        )
    )
    # be careful with factor 2 used in eq (13)

    dpf = []
    for ind, alpha in enumerate(alphas):
        M = (alpha * LB) + L
        dpf_t, info = lgmres(M.tocsr(), B, tol=solver_tolerance)
        dpf_t = ss.zscore(dpf_t)
        dpf.append(dpf_t)

    return dpf

def depth_potential_function_test(mesh, curvature, alphas, equation):
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


    L, LB = compute_mesh_laplacian(mesh, lap_type="fem")

    B = (
        LB
        * (
                curvature -
                ( np.sum(curvature * LB.diagonal()) / np.sum(LB.diagonal()) )
        )
    )
    # be careful with factor 2 used in eq (13)

    dpf = []
    for ind, alpha in enumerate(alphas):
        if equation == 'H':
            M = alpha * LB + L/2
        if equation == 'P':
            M = alpha * LB - L/2
        dpf_t, info = lgmres(M.tocsr(), B, tol=solver_tolerance)
        dpf.append(dpf_t)
    return dpf


def triangle_gradient(mesh, texture_array):
    """
    Compute gradient on a triangular mesh with a scalar function.
    Gradient is computed on each triangle by the function described in
    http://dgd.service.tu-berlin.de/wordpress/vismathws10/2012/10/
    17/gradient-of-scalar-functions/.
    first version author: Guillaume Vicaigne (Internship 2018)
    :param mesh: Triangular mesh
    :param texture_array: Scalar function on Vertices, numpy array
    :return: Gradient on Triangle
    :rtype: Matrix of size number of polygons x 3
    """

    # Initialize Parameters
    vert = mesh.vertices
    poly = mesh.faces
    l_poly = len(poly)
    n = 0
    dicgrad = np.zeros([l_poly, 3])

    # Calculate the Gradient
    for i in range(l_poly):

        # Percentage done
        if int(i / float(l_poly) * 100) > n:
            n += 1
            print(str(n) + " %")
        j = []
        for jj in poly[i]:
            j.append(jj)
        eij = [
            vert[j[1]][0] - vert[j[0]][0],
            vert[j[1]][1] - vert[j[0]][1],
            vert[j[1]][2] - vert[j[0]][2],
        ]
        eki = [
            vert[j[0]][0] - vert[j[2]][0],
            vert[j[0]][1] - vert[j[2]][1],
            vert[j[0]][2] - vert[j[2]][2],
        ]
        ejk = [
            vert[j[2]][0] - vert[j[1]][0],
            vert[j[2]][1] - vert[j[1]][1],
            vert[j[2]][2] - vert[j[1]][2],
        ]
        A = 0.5 * np.linalg.norm(np.cross(eij, ejk))
        N = 0.5 / A * np.cross(ejk, eki)
        dicgrad[i] = np.cross(
            0.5 * N / A,
            np.multiply(texture_array[j[0]], ejk)
            + np.multiply(texture_array[j[1]], eki)
            + np.multiply(texture_array[j[2]], eij),
        )

    return dicgrad


def cross_product(vec1, vec2):
    if vec1.shape != vec2.shape:
        raise Exception("Not the same size")

    res = np.zeros(vec1.shape)
    res[:, 0] = vec1[:, 1] * vec2[:, 2] - vec1[:, 2] * vec2[:, 1]
    res[:, 1] = vec1[:, 2] * vec2[:, 0] - vec1[:, 0] * vec2[:, 2]
    res[:, 2] = vec1[:, 0] * vec2[:, 1] - vec1[:, 1] * vec2[:, 0]

    return res


def gradient_fast(mesh, texture_array):
    """
    Compute gradient on a triangular mesh with a scalar function.
    Gradient is computed on each triangle by the function described in
    http://dgd.service.tu-berlin.de/wordpress/vismathws10/2012/10/
    17/gradient-of-scalar-functions/.
    Formula for the triangle
    grad(f) = - (1/2A) N x (f_i e_{jk} + f_j e_{ik} + f_k e_{ij} )
    And for a mesh
    grad(f) = 1/nb_neighbours * sum(grad_f on each triangle)
    Faster version by using numpy (J Lefevre)
    :param mesh: Triangular mesh
    :param texture_array: Scalar function on Vertices, numpy array
    :return: Gradient on Vertices
    :rtype: numpy.array
    """
    n_tri = mesh.faces.shape[0]
    n_vertex = mesh.vertices.shape[0]
    texture = np.reshape(texture_array, (n_vertex, 1))

    e_ij = mesh.vertices[mesh.faces[:, 1], :] - \
        mesh.vertices[mesh.faces[:, 0], :]
    e_ki = mesh.vertices[mesh.faces[:, 0], :] - \
        mesh.vertices[mesh.faces[:, 2], :]
    e_jk = mesh.vertices[mesh.faces[:, 2], :] - \
        mesh.vertices[mesh.faces[:, 1], :]

    N = cross_product(e_ij, e_jk)
    A = 0.5 * np.linalg.norm(N, 2, 1)
    A = np.reshape(A, (n_tri, 1))
    N = 1 / (2 * A) * N  # may raise an error or be wrong,
    # careful with dims of A and N

    grad_triangle = (
        texture[mesh.faces[:, 0]] * e_jk
        + texture[mesh.faces[:, 1]] * e_ki
        + texture[mesh.faces[:, 2]] * e_ij
    )

    grad_triangle = 1 / (2 * A) * cross_product(N, grad_triangle)

    # From faces to vertices,
    # use the Nvertex x Ntriangles sparse matrix correspondance
    grad_vertex = mesh.faces_sparse * grad_triangle
    grad_vertex = grad_vertex * \
        np.reshape(1 / mesh.vertex_degree, (n_vertex, 1))

    return grad_vertex


def gradient(mesh, texture_array):
    """
    Compute gradient on a triangular mesh with a scalar function.
    Gradient is computed on each triangle by the function described in
    http://dgd.service.tu-berlin.de/wordpress/vismathws10/2012/10/
    17/gradient-of-scalar-functions/.
    On each vertex, compute the mean gradient of all triangle with the vertex.
    first version author: Guillaume Vicaigne (Internship 2018)
    :param mesh: Triangular mesh
    :param texture_array: Scalar function on Vertices, numpy array
    :return: Gradient on Vertices
    :rtype: numpy.array (update 16/12/2020, J Lefevre)
    """

    # Initialize Parameters
    vert = mesh.vertices
    l_vert = len(vert)
    poly = mesh.faces
    l_poly = len(poly)
    n = 0

    # Initialize Dictionnary
    dicgrad = dict()
    for i in range(l_vert):
        dicgrad[i] = [0, 0, 0, 0]

    # Calculate the Gradient, TO DO: avoid the loop and do only array
    # operations
    gradient_vector = np.zeros((l_vert, 3))
    for i in range(l_poly):
        # Percentage done
        if int(i / float(l_poly) * 100) > n:
            n += 1
            print(str(n) + " %")
        j = []
        for jj in poly[i]:
            j.append(jj)
        grad = [0.0, 0.0, 0.0, 0.0]
        eij = [
            vert[j[1]][0] - vert[j[0]][0],
            vert[j[1]][1] - vert[j[0]][1],
            vert[j[1]][2] - vert[j[0]][2],
        ]
        eki = [
            vert[j[0]][0] - vert[j[2]][0],
            vert[j[0]][1] - vert[j[2]][1],
            vert[j[0]][2] - vert[j[2]][2],
        ]
        ejk = [
            vert[j[2]][0] - vert[j[1]][0],
            vert[j[2]][1] - vert[j[1]][1],
            vert[j[2]][2] - vert[j[1]][2],
        ]
        A = 0.5 * np.linalg.norm(np.cross(eij, ejk))
        N = 0.5 / A * np.cross(ejk, eki)
        grad[0:3] = np.cross(
            0.5 * N / A,
            np.multiply(texture_array[j[0]], ejk)
            + np.multiply(texture_array[j[1]], eki)
            + np.multiply(texture_array[j[2]], eij),
        )
        grad[3] = 1.0
        for jj in j:
            dicgrad[jj] = np.add(dicgrad[jj], grad)
    for i in range(l_vert):
        gradient_vector[i] = np.multiply(dicgrad[i][0:3], 1 / dicgrad[i][3])
    return gradient_vector


def operator_grad(mesh):
    idx_list = list()
    neighbors_list = list()
    nbv = len(mesh.vertices)
    adj = stop.ad
    for idx in np.arange(nbv):
        print(idx)
        neighbors = np.where(adj.getcol(idx).toarray() > 0)[0]
        idxs = np.repeat(idx, len(neighbors))
        neighbors_list.append(neighbors)
        idx_list.append(idxs)

        # for neigh in neighbors:
        #    grad_matrix[idx, neigh] = curv[idx] - curv[neigh]

    idx_list = np.hstack(idx_list)
    neighbors_list = np.hstack(neighbors_list)

    diff_curv = curv[idx_list] - curv[neighbors_list]

    grad_matrix = sparse.csr_matrix((diff_curv, (idx_list, neighbors_list)))

def norm_gradient(mesh, texture_array):
    """
    Compute the norm of a vertex Gradient on vertex
    first version author: Guillaume Vicaigne (Internship 2018)
    :param mesh: Triangular mesh
    :param texture_array: Scalar function on Vertices, numpy array
    :return: Gradient's Norm
    """

    # Compute the gradient of the Mesh
    grad = gradient(mesh, texture_array)

    return np.linalg.norm(grad, 2, 1)