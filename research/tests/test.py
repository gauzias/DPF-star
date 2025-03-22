import pymeshlab
import open3d as o3d
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw, ImageFont
import os
from research.tools.data_manager import list_files as dm 
from research.tools.mesh_processing import convert_gii_to_ply as cgp
from research.tools.mesh_processing import convert_ply_to_gii as cpg
from research.tools.mesh_processing import filter_decimation as fd
from research.tools.mesh_processing import snapshot_meshlab as snap

def load_gii_scalars(gii_file):
    """Charge les valeurs scalaires depuis un fichier GIFTI (.gii)"""
    gii_data = nib.load(gii_file)
    scalars = gii_data.darrays[0].data  # Première liste de scalaires
    print(scalars)
    return scalars

def apply_colormap_to_mesh(mesh, scalars, colormap="coolwarm"):
    """Applique une colormap aux sommets du maillage en fonction des scalaires"""
    scalars = np.nan_to_num(scalars)  # Remplace les NaN par zéro si nécessaire
    
    # Normalisation centrée sur 0
    min_val, max_val = np.min(scalars), np.max(scalars)

    #normalized_scalars = (clip_scalars + abs_min)  / (2*coef*abs_min)  # Mise à l'échelle entre 0 et 1
    
    # Appliquer la colormap de Matplotlib
    cmap = plt.get_cmap(colormap)
    colors = cmap(scalars)[:, :3]  # Exclure l'alpha channel
    #colors = cmap(scalars)[:, :3]  # Exclure l'alpha channel

    
    # Assigner les couleurs aux sommets du maillage
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    #return cmap, min_val, max_val
    return cmap, min_val, max_val

def process_ply_with_texture(ply_file, gii_file, output_image, colormap="coolwarm"):
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
    scalars = load_gii_scalars(gii_file)
    cmap, min_val, max_val = apply_colormap_to_mesh(mesh, scalars, colormap)

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

    # Générer la colorbar avec fond blanc et centrée sur 0
    fig, ax = plt.subplots(figsize=(0.5, 5))  # Ajuster la largeur pour éviter la déformation
    norm = plt.Normalize(vmin=min_val, vmax=max_val)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = plt.colorbar(sm, cax=ax, orientation='vertical')
    cb.ax.set_ylabel("DPFstar", rotation=90, labelpad=10)
    cb.ax.yaxis.set_label_position("right")
    cb.ax.tick_params(labelsize=10)
    ax.set_facecolor("white")  # Fond blanc
    colorbar_path = output_image.replace(".png", "_colorbar.png")
    plt.savefig(colorbar_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.close()

    # Fusionner l'image du maillage et la colorbar sur fond blanc
    img_mesh = Image.open(output_image).convert("RGBA")
    img_colorbar = Image.open(colorbar_path).convert("RGBA")
    img_colorbar = img_colorbar.resize((int(img_mesh.width * 0.1), int(img_mesh.height*0.6)))  # Réduction homogène
    new_width = img_mesh.width + img_colorbar.width + 20  # Ajouter un petit espace
    new_height = max(img_mesh.height, img_colorbar.height) + 50  # Ajouter de la place pour le titre
    combined_img = Image.new("RGBA", (new_width, new_height), (255, 255, 255, 255))  # Fond blanc
    combined_img.paste(img_mesh, (0, 0))
    combined_img.paste(img_colorbar, (img_mesh.width + 10, (new_height - img_colorbar.height - 50) // 2))

    # Ajouter le titre du maillage 
    draw = ImageDraw.Draw(combined_img)
    font = ImageFont.truetype("arial.ttf", 24)  # Taille de police augmentée
    text = os.path.basename(ply_file)
    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
    draw.text(((new_width - text_width) // 2, new_height - 50), text, fill="black", font=font)

    combined_img.save(output_image)
    print(f"Image avec colorbar et titre enregistrée : {output_image}")



# path manager and variables
resolutions_list = [100]
data_folder  = "D:/Callisto/data/data_repo_dpfstar/data_test_resolution"
gii_extension = ".gii"
ply_extension = ".ply"


# boucle pour chaque sujet dans la liste : 
try:
    files = dm.list_files(os.path.join(data_folder, "subdataset"))
    print("\n".join(files))
except ValueError as e:
    print(e)

### Loop 

for mesh_path in files :
    # get name file
    mesh_name = dm.get_filename_without_extension(mesh_path, gii_extension)
    print(mesh_name)
    subject_folder = os.path.join(data_folder, mesh_name)
    for res in resolutions_list:
        # snapshot
        name_mesh_decimate = mesh_name + '_res'+ str(res)
        if not os.path.exists(os.path.join(subject_folder, "snapshot")):
            os.makedirs(os.path.join(subject_folder, "snapshot"))

        #output_image = os.path.join(subject_folder, "snapshot", name_mesh_decimate + "_.png")
        output_image = os.path.join(subject_folder, "snapshot", "test.png")
        ply_file = os.path.join(subject_folder, f"{name_mesh_decimate}{ply_extension}")
        gii_file = os.path.join(subject_folder,  "sub-CC00063AN06_ses-15102_hemi-right_desc-drawem_dseg.label.gii")
        process_ply_with_texture(ply_file, gii_file, output_image, colormap="jet") #coolwarm
