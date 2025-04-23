import os
import numpy as np
import nibabel as nib
import open3d as o3d
import pymeshlab
import matplotlib.pyplot as plt
from PIL import Image

from research.tools.mesh_processing import convert_gii_to_ply as cgp


def load_gii_scalars(gii_file):
    gii_data = nib.load(gii_file)
    scalars = gii_data.darrays[0].data
    return scalars


#def apply_colormap_to_mesh(mesh, scalars, colormap):
#    scalars = np.nan_to_num(scalars)
#    cmap = plt.get_cmap(colormap)
#    colors = cmap(scalars)[:, :3]
#    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
#    return cmap

def apply_colormap_to_mesh(mesh, scalars, colormap_name):
    scalars = np.nan_to_num(scalars)

    # Obtenir la colormap (par défaut tab20 a 20 couleurs)
    base_cmap = plt.get_cmap(colormap_name)
    num_colors = base_cmap.N  # typiquement 20 pour tab20

    # Appliquer les couleurs en cyclant sur les labels
    labels = scalars.astype(int)
    repeated_colors = base_cmap(np.arange(num_colors))[:, :3]  # shape (20, 3)

    colors = repeated_colors[labels % num_colors]  # shape (Nb vertices, 3)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    return base_cmap


def set_camera_view(ctr, center, orientation):
    views = {
        "medial": ([1, 0, 0], [0, 1, 0]),
        "lateral": ([-1, 0, 0], [0, -1, 0]),
        "posterior": ([0, -1, 0], [0, 0, 1]),
        "anterior": ([0, 1, 0], [0, 0, 1]),
        "superior": ([0, 0, 1], [0, 1, 0]),
        "inferior": ([0, 0, -1], [0, 1, 0]),
    }

    front, up = views[orientation]
    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_lookat(center)
    ctr.set_zoom(0.7)


def render_and_capture(mesh, edges, center, output_path, orientation):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=800, height=800)
    vis.add_geometry(mesh)
    vis.add_geometry(edges)

    ctr = vis.get_view_control()
    set_camera_view(ctr, center, orientation)

    vis.poll_events()
    vis.update_renderer()

    output_file = os.path.join(output_path, f"{orientation}.png")
    vis.capture_screen_image(output_file)
    vis.destroy_window()
    print(f"[✓] Image enregistrée : {output_file}")


def capture_colored_mesh_snapshots(input_mesh, scalars, output_path, colormap):
    os.makedirs(output_path, exist_ok=True)  # S'assurer que le dossier existe

    # Convert GII to PLY if needed
    if not input_mesh.endswith(".ply"):
        ply_mesh = input_mesh.replace(".gii", ".ply")
        print(f"Conversion GII -> PLY : {ply_mesh}")
        cgp.convert_gii_to_ply(input_mesh, ply_mesh)
        print(f"Conversion terminée : {ply_mesh}")
    else:
        ply_mesh = input_mesh

    # Traitement du mesh
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(ply_mesh)

    processed_ply = os.path.join(output_path, "processed_mesh.ply")
    ms.save_current_mesh(processed_ply)  # Cette ligne pouvait échouer si le dossier n'existait pas

    mesh = o3d.io.read_triangle_mesh(processed_ply)
    mesh.compute_vertex_normals()
    apply_colormap_to_mesh(mesh, scalars, colormap)

    # Bordures pour visualisation
    edges = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    edges.colors = o3d.utility.Vector3dVector([[0, 0, 0] for _ in range(len(edges.lines))])

    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()

    orientations = ["lateral", "medial", "superior", "inferior", "anterior", "posterior"]

    for orientation in orientations:
        render_and_capture(mesh, edges, center, output_path, orientation)


# === Exemple d'utilisation ===
#if __name__ == "__main__":
#    input_mesh = "E:/rel3_dhcp_full/sub-CC00050XX01/ses-7201/anat/sub-CC00050XX01_ses-7201_hemi-left_wm.surf.gii"
#    gii_scalars_file = "E:/research_dpfstar/results_rel3_dhcp/dpfstar/hemi_left/sub-CC00050XX01_ses-7201_hemi-left_wm.surf/dpfstar.gii"
#    scalars = load_gii_scalars(gii_scalars_file)

#    output_path = "E:/snapshot_test/"
#    colormap = "jet"

#    capture_colored_mesh_snapshots(input_mesh, scalars, output_path, colormap)
