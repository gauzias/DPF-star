import slam.curvature as scurv

def normal(mesh):
    """
    :param mesh: Trimesh mesh of interest for computing normal
    :return: [Number of vertices,3] array which [x-direction, y-direction, z-direction] of the normal vertex on each rows
    """
    N = mesh.face_normals
    VertexNormals, Avertex, Acorner, up, vp = scurv.calcvertex_normals(mesh, N)
    return VertexNormals, Avertex, Acorner

def curv(mesh):
    PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = scurv.curvatures_and_derivatives(mesh)
    K1 =  PrincipalCurvatures[0, :]
    K2 =  PrincipalCurvatures[1, :]
    return K1,K2