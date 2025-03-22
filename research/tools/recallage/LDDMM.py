import numpy as np
import open3d as o3d
import pylddmm  # Librairie pour l'alignement élastique
from scipy.spatial import KDTree

def load_ply(filename):
    """Charge un fichier PLY et retourne les sommets et les faces."""
    mesh = o3d.io.read_triangle_mesh(filename)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    return vertices, faces

def save_ply(vertices, faces, output_file):
    """Sauvegarde un maillage au format PLY."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d.io.write_triangle_mesh(output_file, mesh)

def icp_alignment(source_points, target_points, max_iterations=50):
    """Alignement rigide avec ICP."""
    source_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points))
    target_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_points))

    icp_result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, max_correspondence_distance=2.0,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )

    transformed_points = np.asarray(source_pcd.transform(icp_result.transformation).points)
    return transformed_points

def lddmm_deformation(source_points, target_points):
    """Alignement élastique avec LDDMM."""
    tree = KDTree(target_points)
    _, indices = tree.query(source_points, k=1)
    matched_source = source_points[indices]  

    lddmm_solver = pylddmm.LDDMM()  # Initialisation du solveur
    deformed_points = lddmm_solver.register(matched_source, target_points)

    return deformed_points

def compute_average_mesh(mesh_files):
    """Calcule un maillage moyen à partir d'une liste de maillages PLY."""
    all_vertices = []
    ref_vertices, ref_faces = load_ply(mesh_files[0])  # Premier maillage comme référence

    for file in mesh_files:
        vertices, _ = load_ply(file)
        rigid_aligned = icp_alignment(vertices, ref_vertices)  
        elastic_aligned = lddmm_deformation(rigid_aligned, ref_vertices)  
        all_vertices.append(elastic_aligned)

    # Moyenne des coordonnées des sommets
    mean_vertices = np.mean(np.array(all_vertices), axis=0)

    return mean_vertices, ref_faces

# Utilisation avec des maillages PLY de cortex
mesh_list = ["subject1.ply", "subject2.ply", "subject3.ply"]
mean_vertices, mean_faces = compute_average_mesh(mesh_list)
save_ply(mean_vertices, mean_faces, "average_brain_lddmm.ply")

print("Maillage moyen (LDDMM) enregistré sous 'average_brain_lddmm.ply'")
