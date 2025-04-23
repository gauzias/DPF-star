import os
import numpy as np
import nibabel as nib
import open3d as o3d
import pymeshlab
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image
from research.tools.mesh_processing import convert_gii_to_ply as cgp


# ========== Chargement ==========
def load_gii_scalars(gii_file):
    gii_data = nib.load(gii_file)
    scalars = gii_data.darrays[0].data
    return np.nan_to_num(scalars)


# ========== Colormaps ==========
def create_discrete_colormap(value_color_dict):
    values = sorted(value_color_dict.keys())
    colors = [value_color_dict[val] for val in values]

    # Création de bornes de 0.5 autour de chaque valeur
    boundaries = [v - 0.5 for v in values] + [values[-1] + 0.5]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, ncolors=cmap.N)

    return cmap, norm

def apply_colormap_to_mesh(mesh, scalars, cmap, norm=None):
    if norm:
        colors = cmap(norm(scalars))[:, :3]
    else:
        colors = cmap(scalars)[:, :3]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

def apply_named_discrete_colormap_to_mesh(mesh, scalars, colormap_name="tab20"):
    cmap = plt.get_cmap(colormap_name)
    num_colors = cmap.N
    labels = np.nan_to_num(scalars, nan=-1).astype(int)
    colors = np.zeros((len(labels), 3))
    for idx, label in enumerate(labels):
        if label < 0:
            colors[idx] = [0, 0, 0]
        else:
            colors[idx] = cmap(label % num_colors)[:3]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return cmap


# ========== Colorbar ==========
def save_colorbar_image(scalars, cmap, output_path, orientation, is_discrete=False, custom_ticks=None):
    fig, ax = plt.subplots(figsize=(0.8, 6), facecolor="white")
    if is_discrete:
        ticks = custom_ticks
        bounds = np.arange(len(ticks) + 1)

        # étendre le colormap si besoin
        color_cycle = cmap(np.arange(cmap.N))
        repeats = int(np.ceil(len(ticks) / cmap.N))
        extended_colors = np.tile(color_cycle, (repeats, 1))[:len(ticks)]
        extended_cmap = ListedColormap(extended_colors)

        norm = BoundaryNorm(bounds, extended_cmap.N)
        cb = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=extended_cmap),
            cax=ax,
            ticks=np.arange(len(ticks)) + 0.5
        )
        cb.ax.set_yticklabels(ticks)
    else:
        norm = plt.Normalize(vmin=np.min(scalars), vmax=np.max(scalars))
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax)
    cb.ax.tick_params(labelsize=4)
    colorbar_file = os.path.join(output_path, f"colorbar_{orientation}.png")
    plt.savefig(colorbar_file, dpi=150, bbox_inches="tight", transparent=False)
    plt.close()
    return colorbar_file

def combine_with_colorbar(mesh_img_path, colorbar_img_path):
    mesh_img = Image.open(mesh_img_path).convert("RGBA")
    colorbar_img = Image.open(colorbar_img_path).convert("RGBA")
    target_height = mesh_img.height
    colorbar_ratio = colorbar_img.height / colorbar_img.width
    new_width = int(target_height / colorbar_ratio)
    colorbar_img = colorbar_img.resize((new_width, target_height), Image.LANCZOS)
    combined = Image.new("RGBA", (mesh_img.width + new_width, target_height), (255, 255, 255, 0))
    combined.paste(mesh_img, (0, 0))
    combined.paste(colorbar_img, (mesh_img.width, 0))
    combined_path = mesh_img_path.replace(".png", "_with_colorbar.png")
    combined.save(combined_path)
    os.remove(mesh_img_path)
    os.remove(colorbar_img_path)
    print(f"[\u2713] Image combinée : {combined_path}")


# ========== Vue et rendu ==========
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

def render_and_capture(mesh, edges, center, output_path, orientation, colormap_info=None):
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

    if colormap_info is not None:
        cmap, scalars, is_discrete, tick_labels = colormap_info
        colorbar_img = save_colorbar_image(scalars, cmap, output_path, orientation, is_discrete, tick_labels)
        combine_with_colorbar(output_file, colorbar_img)


# ========== Traitement principal ==========
def capture_colored_mesh_snapshots(input_mesh, scalars, output_path, colormap_type, colormap, custom_dict=None):
    os.makedirs(output_path, exist_ok=True)
    if not input_mesh.endswith(".ply"):
        ply_mesh = input_mesh.replace(".gii", ".ply")
        print(f"Conversion GII -> PLY : {ply_mesh}")
        cgp.convert_gii_to_ply(input_mesh, ply_mesh)
    else:
        ply_mesh = input_mesh

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(ply_mesh)
    processed_ply = os.path.join(output_path, "processed_mesh.ply")
    ms.save_current_mesh(processed_ply)

    mesh = o3d.io.read_triangle_mesh(processed_ply)
    mesh.compute_vertex_normals()

    colormap_info = None

    if colormap_type == "continuous":
        cmap = plt.get_cmap(colormap)
        apply_colormap_to_mesh(mesh, scalars, cmap)
        colormap_info = (cmap, scalars, False, None)
    elif colormap_type == "discrete":
        cmap = apply_named_discrete_colormap_to_mesh(mesh, scalars, colormap)
        colormap_info = (cmap, scalars, True, list(np.unique(scalars[scalars >= 0])))
    elif colormap_type == "custom":
        if not custom_dict:
            raise ValueError("Une colormap personnalisée nécessite un dictionnaire de couleurs.")
        cmap, norm = create_discrete_colormap(custom_dict)
        apply_colormap_to_mesh(mesh, scalars, cmap, norm)
        colormap_info = (cmap, scalars, True, list(custom_dict.keys()))
    else:
        raise ValueError(f"Type de colormap inconnu : {colormap_type}")

    edges = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    edges.colors = o3d.utility.Vector3dVector([[0, 0, 0]] * len(edges.lines))

    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()

    orientations = ["lateral", "medial", "superior", "inferior", "anterior", "posterior"]
    for orientation in orientations:
        render_and_capture(mesh, edges, center, output_path, orientation, colormap_info)
