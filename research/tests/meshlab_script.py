import pymeshlab
import os

def process_ply_with_pymeshlab(ply_file, output_image):
    # Charger le maillage
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(ply_file)

    # Appliquer un filtre pour marquer les arêtes du maillage
    ms.apply_filter("compute_texcoord_parametrization_triangle_trivial_per_wedge")  # Marque les arêtes
    #ms.apply_filter("invert_faces_normals")  # Permet de mieux voir les arêtes

    # Sauvegarder une capture d’écran
    ms.save_snapshot(imagefilename =output_image)

    print(f"Image enregistrée : {output_image}")

# Modifier les chemins ici
folder = "D:\Callisto\data\data_repo_dpfstar\data_test_resolution\sub-CC00063AN06_ses-15102_hemi-L_space-T2w_wm.surf"
ply_file = os.path.join(folder, "sub-CC00063AN06_ses-15102_hemi-L_space-T2w_wm.surf_res25.ply")  # Chemin du fichier PLY
if not os.path.exists(os.path.join(folder, "snapshot")):
    os.makedirs(os.path.join(folder, "snapshot"))
output_image = os.path.join(folder, "snapshot", "sub-CC00063AN06_ses-15102_hemi-L_space-T2w_wm.surf_res25.png")  # Chemin de l'image de sortie

process_ply_with_pymeshlab(ply_file, output_image)