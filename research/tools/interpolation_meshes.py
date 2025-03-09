import nibabel as nib
import numpy as np
import trimesh
from scipy.spatial import KDTree

def load_gii_mesh(gii_file):
    """Charge un fichier GIFTI (.gii) et retourne les sommets, faces et la texture"""
    gii = nib.load(gii_file)
    vertices = gii.darrays[0].data  # Sommets (Nx3)
    faces = gii.darrays[1].data  # Faces (Mx3)
    if len(gii.darrays) > 2:
        texture = gii.darrays[2].data  # Valeur scalaire par sommet
    else:
        texture = None
    return vertices, faces, texture

def find_nearest_face(mesh_high, points_low):
    """Trouve la face du maillage haute résolution la plus proche pour chaque point du maillage basse résolution"""
    tree = KDTree(mesh_high.vertices)  # KD-Tree pour la recherche rapide
    nearest_faces = []
    for point in points_low:
        _, nearest_vertex_idx = tree.query(point)
        nearest_face_idx = np.where(mesh_high.faces == nearest_vertex_idx)[0]
        if len(nearest_face_idx) > 0:
            nearest_faces.append(mesh_high.faces[nearest_face_idx[0]])
        else:
            nearest_faces.append(None)
    return nearest_faces

def barycentric_interpolation(v0, v1, v2, p, f_values):
    """Interpole la valeur en p en utilisant les coordonnées barycentriques"""
    # Calcul des coordonnées barycentriques
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = p - v0
    d00 = np.dot(v0v1, v0v1)
    d01 = np.dot(v0v1, v0v2)
    d11 = np.dot(v0v2, v0v2)
    d20 = np.dot(v0p, v0v1)
    d21 = np.dot(v0p, v0v2)
    denom = d00 * d11 - d01 * d01

    if denom == 0:
        return np.mean(f_values)  # Valeur moyenne en cas de problème numérique

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w

    return u * f_values[0] + v * f_values[1] + w * f_values[2]

def map_low_to_high(mesh_high, mesh_low, depth_high):
    """
    Projette la profondeur du maillage haute résolution sur le maillage basse résolution.
    
    Args:
        mesh_high: Maillage haute résolution avec sommets et faces.
        mesh_low: Maillage basse résolution avec sommets.
        depth_high: Liste des profondeurs associées aux sommets de mesh_high (taille N, où N est le nombre de sommets).

    Returns:
        mapped_depth: Profondeur interpolée sur les sommets de mesh_low.
    """
    mapped_depth = np.zeros(len(mesh_low.vertices))
    nearest_faces = find_nearest_face(mesh_high, mesh_low.vertices)
    print(nearest_faces)
    
    for i, face in enumerate(nearest_faces):
        print("loop:", i)
        print("nearest face", face)
        if face is not None:
            v0, v1, v2 = mesh_high.vertices[face]
            print("v0, v1, v2", v0, v1, v2)
            f_values = depth_high[np.array(face)]

            print("f_values:", f_values)
            mapped_depth[i] = barycentric_interpolation(v0, v1, v2, mesh_low.vertices[i], f_values)
        else:
            mapped_depth[i] = np.nan  # Valeur manquante
    
    return mapped_depth

