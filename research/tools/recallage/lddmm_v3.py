import numpy as np
import open3d as o3d
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import KDTree
from joblib import Parallel, delayed
import os


def project_texture_to_subject(mean_texture, subject_vertices, mean_vertices, mapping_file):
    """Projette une texture du cerveau moyen vers un sujet."""
    indices = np.load(mapping_file)  # Charger le mapping
    subject_texture = mean_texture[indices]  # Associer la texture aux indices correspondants
    return subject_texture


def remap_all_to_reference(all_vertices, ref_vertices):
    """Remappe tous les maillages sur la grille de référence pour garantir une forme homogène."""
    print("🔹 Vérification et remapping des sommets...")

    fixed_all_vertices = []
    for i, vertices in enumerate(all_vertices):
        if vertices.shape[0] != ref_vertices.shape[0]:
            print(f"⚠️ Remapping nécessaire pour le sujet {i} : {vertices.shape[0]} → {ref_vertices.shape[0]}")
            tree = KDTree(vertices)
            _, indices = tree.query(ref_vertices, k=1)
            vertices = vertices[indices]  # Remap sur ref_vertices
        
        fixed_all_vertices.append(vertices)

    return np.array(fixed_all_vertices)  # Retourne un tableau homogène


def compute_subject_to_average_mapping(all_vertices, mean_vertices, output_folder):
    """Calcule et sauvegarde le mapping entre chaque sujet et le maillage moyen."""
    print("🔹 Calcul du mapping entre chaque sujet et le cerveau moyen...")

    for i, subject_vertices in enumerate(all_vertices):
        tree = KDTree(mean_vertices)
        _, indices = tree.query(subject_vertices, k=1)  # Trouve le sommet le plus proche dans le cerveau moyen
        
        mapping_file = os.path.join(output_folder, f"mapping_subject_{i}.npy")
        np.save(mapping_file, indices)  # Sauvegarde des indices
        
        print(f"✅ Mapping sauvegardé pour le sujet {i}: {mapping_file}")


def normalize_scale(vertices):
    """Normalise la taille du maillage en le ramenant à une échelle commune."""
    max_dist = np.max(np.linalg.norm(vertices - np.mean(vertices, axis=0), axis=1))
    return vertices / max_dist  # Réduit tous les maillages à une échelle commune


def compute_symmetric_average(all_vertices):
    """Calcule une moyenne équilibrée en recentrant chaque maillage."""
    mean_vertices = np.mean(np.array(all_vertices), axis=0)

    # Centrer chaque maillage sur la moyenne avant de moyenner
    new_aligned = []
    for vertices in all_vertices:
        translation = mean_vertices - np.mean(vertices, axis=0)  # Décalage pour centrer
        new_aligned.append(vertices + translation)  # Appliquer translation

    return np.mean(np.array(new_aligned), axis=0)  # Nouvelle moyenne recentrée



# 📌 Réduction du maillage
def simplify_mesh(vertices, faces, reduction_ratio=0.5):
    """Réduit la taille du maillage pour accélérer les calculs."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    print(f"🔹 Simplification du maillage (réduction {reduction_ratio*100:.0f}%)...")
    mesh = mesh.simplify_quadric_decimation(int(len(vertices) * reduction_ratio))
    
    simplified_vertices = np.asarray(mesh.vertices)
    simplified_faces = np.asarray(mesh.triangles)

    print(f"✅ Maillage réduit : {len(vertices)} → {len(simplified_vertices)} sommets")
    return simplified_vertices, simplified_faces

# 📌 Chargement des maillages PLY
def load_ply(filename, reduction_ratio=0.5):
    """Charge un fichier PLY et applique une simplification si nécessaire."""
    mesh = o3d.io.read_triangle_mesh(filename)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    if reduction_ratio < 1.0:
        vertices, faces = simplify_mesh(vertices, faces, reduction_ratio)

    return vertices, faces

# 📌 Sauvegarde du maillage
def save_ply(vertices, faces, output_file):
    """Sauvegarde un maillage au format PLY."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d.io.write_triangle_mesh(output_file, mesh)

# 📌 Alignement rigide optimisé avec sous-échantillonnage
def icp_alignment(source_points, target_points, max_iterations=30, downsample_ratio=0.5):
    """Alignement rigide avec ICP optimisé."""
    print(f"🔹 Sous-échantillonnage de {downsample_ratio*100:.0f}% pour l’ICP...")
    
    source_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points))
    target_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_points))

    # Échantillonnage pour accélérer l'ICP
    source_pcd = source_pcd.voxel_down_sample(voxel_size=0.5)
    target_pcd = target_pcd.voxel_down_sample(voxel_size=0.5)

    icp_result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, max_correspondence_distance=1.0,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )

    transformed_points = np.asarray(source_pcd.transform(icp_result.transformation).points)
    print("✅ ICP terminé")
    return transformed_points

# 📌 Déformation élastique avec interpolation linéaire
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

    # **🔍 Vérification des NaN après l'interpolation**
    if np.isnan(deformed_points).any() or np.isinf(deformed_points).any():
        print("⚠️ Erreur : Des NaN/Inf détectés après interpolation, correction...")
        deformed_points = np.nan_to_num(deformed_points)  # Remplace NaN/Inf par 0

    print(f"✅ Interpolation réussie : {deformed_points.shape}")
    return deformed_points


# 📌 Remapping des sommets pour garantir la même taille
def remap_to_reference(deformed_points, ref_vertices):
    """Remappe les sommets interpolés sur la grille de référence."""
    if np.isnan(deformed_points).any() or np.isinf(deformed_points).any():
        print("⚠️ Erreur : Des NaN ou Inf ont été détectés dans deformed_points !")
        print("🔹 Remplacement des NaN par les valeurs originales...")
        deformed_points = np.nan_to_num(deformed_points)  # Remplace NaN/Inf par 0
    
    if deformed_points.shape[0] != ref_vertices.shape[0]:
        print(f"⚠️ Remapping nécessaire : {deformed_points.shape[0]} → {ref_vertices.shape[0]} sommets")
        tree = KDTree(deformed_points)
        _, indices = tree.query(ref_vertices, k=1)
        deformed_points = deformed_points[indices]  # Remap sur ref_vertices
    
    return deformed_points

# 📌 Traitement d’un sujet (alignement + déformation + remapping) → Exécutable en parallèle
def process_subject(file, ref_vertices):
    """Traite un sujet en parallèle : alignement + déformation + remapping."""
    vertices, _ = load_ply(file, reduction_ratio=0.5)
    rigid_aligned = icp_alignment(vertices, ref_vertices)
    elastic_aligned = linear_interpolation_deformation(rigid_aligned, ref_vertices)
    
    # **Remap sur la grille de référence**
    remapped_vertices = remap_to_reference(elastic_aligned, ref_vertices)
    return remapped_vertices

# 📌 Calcul du maillage moyen en parallèle
def compute_average_mesh_parallel(mesh_files, output_folder, n_jobs=-1):
    """Calcule le maillage moyen et sauvegarde les mappings vers les sujets."""
    ref_vertices, ref_faces = load_ply(mesh_files[0], reduction_ratio=0.5)

    print(f"🔹 Traitement de {len(mesh_files)} maillages en parallèle...")

    all_vertices = Parallel(n_jobs=n_jobs)(
        delayed(process_subject)(file, ref_vertices) for file in mesh_files
    )

    # **Remap tous les maillages sur la grille de référence**
    all_vertices = remap_all_to_reference(all_vertices, ref_vertices)

    # **Conversion en tableau NumPy homogène**
    all_vertices = np.array(all_vertices)

    # **Calcul du maillage moyen**
    mean_vertices = np.mean(all_vertices, axis=0)

    print(f"✅ Maillage moyen recalculé ({mean_vertices.shape[0]} sommets)")

    # **📌 Sauvegarde des mappings**
    compute_subject_to_average_mapping(all_vertices, mean_vertices, output_folder)

    return mean_vertices, ref_faces



# 📌 Dossier contenant les maillages
folder = "D:/Callisto/data/data_repo_dpfstar/data_test_resolution/dataset_lddmm"

sub1 = os.path.join(folder, "sub-KKI2009_113_ses-MR1_hemi-L_space-T2w_wm.surf_res50.ply")
sub2 = os.path.join(folder, "sub-KKI2009_142_ses-MR1_hemi-L_space-T2w_wm.surf_res50.ply")
sub3 = os.path.join(folder, "sub-KKI2009_505_ses-MR1_hemi-L_space-T2w_wm.surf_res50.ply")

mesh_list = [sub1, sub2, sub3]
mean_vertices, mean_faces = compute_average_mesh_parallel(mesh_list, folder)

# 📌 Sauvegarde du résultat final
output_file = os.path.join(folder, "average_brain_parallel.ply")
save_ply(mean_vertices, mean_faces, output_file)

print(f"✅ Maillage moyen optimisé enregistré sous '{output_file}'")
