import numpy as np
import open3d as o3d
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import KDTree
import os

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

def linear_interpolation_deformation(source_points, target_points):
    """Déformation élastique avec LinearNDInterpolator."""
    print(f"📌 Début de l'interpolation : {source_points.shape} -> {target_points.shape}")

    tree = KDTree(source_points)
    _, indices = tree.query(target_points, k=1)
    control_points = source_points[indices]  # Correspondance des sommets

    print(f"🔍 KDTree : {control_points.shape}")

    if control_points.shape[0] == 0:
        print("⚠️ Erreur : Aucun point de contrôle trouvé !")
        return source_points  # Retourner les points inchangés en cas d'échec

    # Déformation avec interpolation linéaire
    interp_x = LinearNDInterpolator(control_points, target_points[:, 0])
    interp_y = LinearNDInterpolator(control_points, target_points[:, 1])
    interp_z = LinearNDInterpolator(control_points, target_points[:, 2])

    deformed_points = np.column_stack((
        interp_x(source_points),
        interp_y(source_points),
        interp_z(source_points)
    ))

    print(f"✅ Interpolation réussie : {deformed_points.shape}")
    return deformed_points

def compute_average_mesh(mesh_files):
    """Calcule un maillage moyen en appliquant une interpolation linéaire."""
    all_vertices = []
    ref_vertices, ref_faces = load_ply(mesh_files[0])  # Premier maillage comme référence

    print(f"🔹 Référence : {mesh_files[0]} avec {len(ref_vertices)} sommets et {len(ref_faces)} faces.")

    for i, file in enumerate(mesh_files):
        vertices, _ = load_ply(file)
        print(f"📂 Chargement : {file} ({len(vertices)} sommets)")

        rigid_aligned = icp_alignment(vertices, ref_vertices)
        print(f"🔄 Après ICP (alignement rigide) : {rigid_aligned.shape}")

        elastic_aligned = linear_interpolation_deformation(rigid_aligned, ref_vertices)
        print(f"💠 Après interpolation linéaire : {elastic_aligned.shape}")

        all_vertices.append(elastic_aligned)

    if not all_vertices:
        print("⚠️ Erreur : Liste all_vertices est vide !")
        return ref_vertices, ref_faces  # Retourne la référence pour éviter un crash

    mean_vertices = np.mean(np.array(all_vertices), axis=0)
    print(f"✅ Maillage moyen calculé ({mean_vertices.shape[0]} sommets)")

    return mean_vertices, ref_faces

# 📌 Dossier contenant les maillages
folder = "D:/Callisto/data/data_repo_dpfstar/data_test_resolution/dataset_lddmm"
sub1 = os.path.join(folder, "sub-KKI2009_113_ses-MR1_hemi-L_space-T2w_wm.surf_res50.ply")
sub2 = os.path.join(folder, "sub-KKI2009_142_ses-MR1_hemi-L_space-T2w_wm.surf_res50.ply")
sub3 = os.path.join(folder, "sub-KKI2009_505_ses-MR1_hemi-L_space-T2w_wm.surf_res50.ply")

mesh_list = [sub1, sub2, sub3]
mean_vertices, mean_faces = compute_average_mesh(mesh_list)
save_ply(mean_vertices, mean_faces, os.path.join(folder, "average_brain_linear.ply"))

print("✅ Maillage moyen (LinearNDInterpolator) enregistré sous 'average_brain_linear.ply'")
