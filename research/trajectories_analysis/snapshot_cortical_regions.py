import pymeshlab
import open3d as o3d
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw, ImageFont
import os
import pandas as pd
from research.tools.data_manager import list_files as dm 
from research.tools.mesh_processing import convert_gii_to_ply as cgp
from research.tools.mesh_processing import convert_ply_to_gii as cpg
from research.tools.mesh_processing import filter_decimation as fd
from research.tools.mesh_processing import snapshot_meshlab as snap

def load_gii_scalars(gii_file):
    """Charge les valeurs scalaires depuis un fichier GIFTI (.gii)"""
    gii_data = nib.load(gii_file)
    scalars = gii_data.darrays[0].data  # Première liste de scalaires
    return scalars

def apply_colormap_to_mesh(mesh, scalars, colormap):
    """Applique une colormap aux sommets du maillage en fonction des scalaires"""
    scalars = np.nan_to_num(scalars)  # Remplace les NaN par zéro si nécessaire
    cmap = plt.get_cmap(colormap)
    colors = cmap(scalars)[:, :3]  # Exclure l'alpha channel
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    return cmap

def process_ply_with_texture(ply_file, scalars, output_image, colormap):
    """Charge le maillage, applique la texture et affiche le rendu avec Open3D"""
    
    # Charger le maillage avec PyMeshLab
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(ply_file)
    
    # Sauvegarder le maillage transformé
    processed_ply = os.path.join(os.path.dirname(output_image), "processed_mesh.ply")
    ms.save_current_mesh(processed_ply)
    
    print(f"Maillage transformé enregistré : {processed_ply}")

    # Charger le maillage avec Open3D
    mesh = o3d.io.read_triangle_mesh(processed_ply)
    mesh.compute_vertex_normals()

    # Charger les valeurs scalaires du fichier GII et appliquer la colormap
    
    cmap = apply_colormap_to_mesh(mesh, scalars, colormap)

    # Appliquer une rotation de 90° autour de l'axe Y et Z
    RY = mesh.get_rotation_matrix_from_axis_angle([0, -np.pi / 2, 0])  # 90° en radians autour de Y
    mesh.rotate(RY, center=(0, 0, 0))
    RZ = mesh.get_rotation_matrix_from_axis_angle([0, 0, -np.pi / 2])  # 90° en radians autour de Z
    mesh.rotate(RZ, center=(0, 0, 0))

    # Générer les arêtes en wireframe (trait noir fin)
    edges = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    edges.colors = o3d.utility.Vector3dVector([[0, 0, 0] for _ in range(len(edges.lines))])  # Noir

    # Centrer et ajuster la vue
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()

    # Visualisation avec Open3D
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=800, height=800)

    vis.add_geometry(mesh)
    vis.add_geometry(edges)

    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_lookat(center)
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.7)

    vis.poll_events()
    vis.update_renderer()

    # Capturer une image
    vis.capture_screen_image(output_image)
    vis.destroy_window()

    print(f"Image enregistrée : {output_image}")

    # Fusionner l'image du maillage et la colorbar sur fond blanc
    img_mesh = Image.open(output_image).convert("RGBA")
    img_mesh.save(output_image)




# Dossiers
folder_cortical_metric_left = "E:\\research_dpfstar\\results_rel3_dhcp\\cortical_metrics_sulc\\hemi_left"

    # Chargement
df_all_left = pd.DataFrame()

cortical_metric_left = os.path.join(folder_cortical_metric_left, "sub-CC00180XX08_ses-59500_cortical_metrics.csv")
df_all_left = pd.read_csv(cortical_metric_left)

labels = df_all_left["Label"].values
rois = df_all_left["ROI"].values

for i, label in enumerate(labels):
    roi = rois[i]
    print(label)
    output_image = os.path.join("E:", "snapshot", roi + ".png")
    gii_file = "E:\\rel3_dhcp_full\\sub-CC00180XX08\\ses-59500\\anat\\sub-CC00180XX08_ses-59500_hemi-left_desc-drawem_dseg.label.gii"
    scalars = load_gii_scalars(gii_file)
    scalars_label = np.array((scalars == label).astype(int)) * label
    print(np.unique(scalars_label))
    path_input_mesh = "E:\\rel3_dhcp_full\\sub-CC00180XX08\\ses-59500\\anat\\sub-CC00180XX08_ses-59500_hemi-left_wm.surf.gii"
    path_ply_mesh = "E:\\rel3_dhcp_full\\sub-CC00180XX08\\ses-59500\\anat\\sub-CC00180XX08_ses-59500_hemi-left_wm.surf.ply"


    ply_mesh = cgp.convert_gii_to_ply(path_input_mesh, path_ply_mesh)
    
    
    process_ply_with_texture(path_ply_mesh, scalars_label, output_image, colormap="Paired") 