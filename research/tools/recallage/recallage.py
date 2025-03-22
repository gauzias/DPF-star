import nibabel as nib
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import vedo
from pycpd import DeformableRegistration

# Charger le maillage du sujet
subject_mesh = nib.load("subject.pial.gii")
subject_vertices = subject_mesh.darrays[0].data
subject_faces = subject_mesh.darrays[1].data

# Charger l'atlas (fsaverage)
atlas_mesh = nib.load("fsaverage.pial.gii")
atlas_vertices = atlas_mesh.darrays[0].data
atlas_faces = atlas_mesh.darrays[1].data



def apply_icp(source_pts, target_pts):
    # Convertir en nuage de points Open3D
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_pts)

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_pts)

    # Appliquer ICP
    threshold = 10.0  # Distance max
    trans_init = np.eye(4)  # Transformation initiale
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return reg_p2p.transformation

# Appliquer l'alignement ICP
transformation = apply_icp(atlas_vertices, subject_vertices)

# Transformer l'atlas
atlas_vertices_transformed = np.dot(atlas_vertices, transformation[:3, :3].T) + transformation[:3, 3]



def deform_registration(source, target):
    reg = DeformableRegistration(X=source, Y=target)
    TY, _ = reg.register()
    return TY

# Appliquer CPD pour une transformation non rigide
atlas_vertices_deformed = deform_registration(atlas_vertices_transformed, subject_vertices)



# Charger les labels de l'atlas (Desikan-Killiany)
atlas_labels = np.load("fsaverage_labels.npy")  # Fichier contenant les étiquettes Desikan-Killiany

# Construire un arbre KD pour la recherche rapide des voisins
tree = cKDTree(atlas_vertices_deformed)
_, indices = tree.query(subject_vertices, k=1)  # Trouver le sommet le plus proche dans l'atlas

# Attribuer les labels correspondants
subject_labels = atlas_labels[indices]


# Créer un nouveau fichier Gifti avec les labels attribués
new_gifti = nib.GiftiImage()
new_gifti.add_gifti_data_array(nib.gifti.GiftiDataArray(subject_labels.astype(np.int32)))
nib.save(new_gifti, "subject_atlas.gii")




# Créer les objets 3D
subject_mesh_viz = vedo.Mesh([subject_vertices, subject_faces])
subject_mesh_viz.cmap("jet", subject_labels)  # Coloration avec l'atlas

# Affichage
vedo.show(subject_mesh_viz, axes=1)


