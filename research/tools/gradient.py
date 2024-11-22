import numpy as np
import slam.differential_geometry as sdg
import math
import os
import slam.io as sio
import pandas as pd
import pickle

def gradient_coord_line(coord_line):
    """
    :param coord_line: [nb,3] array with the ordered coordinates of vertex connected each other trough a line
    :return: gdt_line : [nb,3] array : first order derivation according x, y and z at each vertex.
    The gradient starts from last vertex of the line and ends to the first vertex of the line. So gradient is not
    computed for the first vertex and we add a value of 0 instead.
    """
    nb_vertices = len(coord_line)
    gdt_line = [ coord_line[i] - coord_line[i+1] for i in np.arange(0,nb_vertices-1)]
    gdt_line.insert(0,gdt_line[0])
    gdt_line = np.array(gdt_line)
    return gdt_line

def gradient_texture(texture, mesh):
    """
    :param texture: texture
    :param mesh: mesh correponding to the texture
    :return: grad : [nb vertex, 3] array with x-direction, ydirection and zdirection of gradient at each point od the mesh
    """
    texture = np.array(texture)
    if texture.ndim ==1:
        texture = np.reshape(texture, (1, len(texture)))
    grad = list()
    nb_layer = len(texture)
    for i in np.arange(nb_layer):
        grad_i = sdg.gradient(mesh, texture[i])
        grad.append(grad_i)
    if len(grad)==1:
        grad = grad[0]
    return grad


def gradient_depth(folder_meshes, folder_subject_analysis, sub, ses, method):
    """
    This fonction is specific to the folder organisation of this project
    :param folder_meshes: PATH
    :param folder_subject_analysis: PATH
    :param sub: STRING
    :param ses: STRING
    :param method:  STRING 'dpfstar' or 'dpf' or 'curv' or 'sulc'
    :return: save gradient depth in a dataframe
    """
    mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-L_space-T2w_wm.surf.gii'
    mesh_path = os.path.join(folder_meshes, mesh_name)
    mesh = sio.load_mesh(mesh_path)
    # for dpfstar or dpf
    if method == "dpfstar" :
        print('method: ', method)
        alphas_path = os.path.join(folder_subject_analysis, sub + '_' + ses, 'surface_processing', method,
                                    'alphas.pkl')
        with open(alphas_path, 'rb') as file_alphas:
            alphas = pickle.load(file_alphas)
        print(alphas)
        depth_path = os.path.join(folder_subject_analysis, sub + '_' + ses, 'surface_processing', method,
                                    sub + '_' + ses + '_' + method + '.gii')
        depth = sio.load_texture(depth_path).darray
        grad_depth = gradient_texture(depth, mesh)
        grad_depth = np.array(grad_depth)
        nb_alpha = grad_depth.shape[0]
        nb_vtx = grad_depth.shape[1]
        nb_dim = grad_depth.shape[2]
        grad_depth = grad_depth.reshape(nb_alpha * nb_vtx, nb_dim)
        df_grad = pd.DataFrame(grad_depth, columns=['x', 'y', 'z'])
        df_grad['alpha'] = np.repeat(alphas, nb_vtx)
        save_folder = os.path.join(folder_subject_analysis, sub + '_' + ses, 'surface_processing', method, 'derivatives','gradient')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        df_grad.to_csv(os.path.join(save_folder, sub + '_' + ses + '_grad_' + method + '.csv'), index=False)
    if method == "dpf":
        print('method: ', method)
        depth_path = os.path.join(folder_subject_analysis, sub + '_' + ses, 'surface_processing', method,
                                  sub + '_' + ses + '_' + method + '.gii')
        depth = sio.load_texture(depth_path).darray[13]
        grad_depth = gradient_texture(depth, mesh)
        df_grad_depth = pd.DataFrame(grad_depth, columns=['x', 'y', 'z'])
        folder_grad_depth = os.path.join(folder_subject_analysis, sub + '_' + ses, 'surface_processing', method,
                                         'derivatives', 'gradient')
        if not os.path.exists(folder_grad_depth):
            os.makedirs(folder_grad_depth)
        df_grad_depth.to_csv(os.path.join(folder_grad_depth, sub + '_' + ses + '_grad_' + method + '.csv'), index=False)
    # for curv or sulc
    if (method == "curv") | (method == "sulc"):
        print('method: ', method)
        if method == "sulc":
            depth_path = os.path.join(folder_subject_analysis, sub + '_' + ses, 'surface_processing', 'sulc',
                                 'sub-' + sub + '_ses-' + ses + '_hemi-L_space-T2w_sulc.shape.gii')
            depth = sio.load_texture(depth_path).darray[0]
        if method == "curv":
            K1_path = os.path.join(folder_subject_analysis, sub + '_' + ses, 'surface_processing', 'curvature',
                               sub + '_' + ses + '_K1.gii')
            K2_path = os.path.join(folder_subject_analysis, sub + '_' + ses, 'surface_processing', 'curvature',
                               sub + '_' + ses + '_K2.gii')
            K1 = sio.load_texture(K1_path).darray[0]
            K2 = sio.load_texture(K2_path).darray[0]
            depth = 0.5 * (K1 + K2)

        grad_depth = gradient_texture(depth, mesh)
        df_grad_depth = pd.DataFrame(grad_depth, columns=['x', 'y', 'z'])
        folder_grad_depth = os.path.join(folder_subject_analysis, sub + '_' + ses, 'surface_processing', method,
                                        'derivatives', 'gradient')
        if not os.path.exists(folder_grad_depth):
            os.makedirs(folder_grad_depth)
        df_grad_depth.to_csv(os.path.join(folder_grad_depth, sub + '_' + ses + '_grad_' + method +'.csv'), index=False)


def make_arrow(df_grad, nb_vert, vertices):
    """
    function for visbrain visualisation of vectors on mesh
    :param df_grad:
    :param nb_vert:
    :param vertices:
    :return: arrows_grad
    """
    dt_grad = np.dtype([('start', float, 3), ('end', float, 3)])
    df_grad = np.array(df_grad)
    grad_norm = np.sqrt(np.sum(np.square(df_grad), axis=1))
    df_grad = 0.5 * (df_grad / np.transpose(np.vstack([grad_norm, grad_norm, grad_norm])))
    arrows_grad = np.zeros(nb_vert, dtype=dt_grad)
    arrows_grad['start'] = vertices
    arrows_grad['end'] = np.array(vertices) + np.array(df_grad)
    return arrows_grad



def projection_vector_plane(vectors, normals):
    """
    :param vectors: list of vectors
    :param normals: list of normals that define the plans on which you want project the vectors
    :return: proj_vector : the list of vector projected respectively to the list of plans
    """
    proj_vectors = [vec - np.dot(vec, nml) * nml for (vec, nml) in zip(vectors, normals)]
    proj_vectors = np.array(proj_vectors)
    return proj_vectors

def vector_product(uu, normals):
    """
    :param uu: list of vectors
    :param normals: list of  vectors
    :return: vv,list of the cross product between uu and the normals (element wise)
    """
    vv = [ np.cross(nml,pvl) for (nml, pvl) in zip( normals, uu)]
    vv = np.array(vv)
    return vv


def get_2D_coord(x,u,v):
    """
    for a given  vector u, vector v and vector x, (in the same plan), express x = a*u + b*v and return (a,b)
    :param x: 3D vector to convert
    :param u: 3D basis vector
    :param v: 3D basis vector
    :return: a, b , x = au + bv
    """
    a = np.dot(x,u) / np.dot(u,u)
    b = np.dot(x,v) / np.dot(v,v)

    no = np.sqrt(np.square(a) + np.square(b))
    a = a / no
    b = b /no
    return [a,b]


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in degree between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return math.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
