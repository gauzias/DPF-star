import os
import numpy as np
from numba import njit
from research.tools import rw
from research.tools import topology as slt
from research.tools import texture as stex
from research.tools.snapshot import snap_mesh  # 📸 capture des screenshots

# === Paramètres ===
#subject_id = "CC00666XX15" 
#session_id = "198200"
subject_id = "CC00804XX12"
session_id = "900"
hemi = "left"

data_root = "E:/rel3_dHCP_full"
mesh_path = os.path.join(data_root, f"sub-{subject_id}/ses-{session_id}/anat", f"sub-{subject_id}_ses-{session_id}_hemi-{hemi}_wm.surf.gii")
dpf_smooth_path = f"E:/research_dpfstar/results_rel3_dhcp/dpfstar_smooth/hemi_{hemi}/sub-{subject_id}_ses-{session_id}_hemi-{hemi}_dpfstar_smooth.gii"
output_folder = f"E:/research_dpfstar/results_rel3_dhcp/watershed_labels_opt/hemi_{hemi}"
os.makedirs(output_folder, exist_ok=True)

# === Chargement ===
print("[INFO] Chargement mesh et profondeur...")
mesh = rw.load_mesh(mesh_path)
depth = rw.read_gii_file(dpf_smooth_path).flatten()
adj_matrix = slt.adjacency_matrix(mesh)

# === Liste des voisins ===
print("[INFO] Construction de la liste des voisins...")
nb_vertices = mesh.vertices.shape[0]
neighbors = [adj_matrix.indices[adj_matrix.indptr[i]:adj_matrix.indptr[i+1]] for i in range(nb_vertices)]

# === Watershed avec numba ===
@njit
def run_watershed(depth, neighbors):
    nb_vertices = depth.shape[0]
    labels = -1 * np.ones(nb_vertices, dtype=np.int32)
    sorted_idx = np.argsort(depth)
    current_label = 0

    for idx in sorted_idx:
        labeled_neighs = []
        for n in neighbors[idx]:
            if labels[n] != -1:
                labeled_neighs.append(labels[n])
        
        if len(labeled_neighs) == 0:
            labels[idx] = current_label
            current_label += 1
        else:
            labels[idx] = labeled_neighs[0]
    return labels

print("[INFO] Lancement du Watershed optimisé...")
labels_array = run_watershed(depth, neighbors)
n_regions = np.max(labels_array) + 1
print(f"[✓] Watershed terminé avec {n_regions} régions.")

# === Sauvegarde ===
gii_out = os.path.join(output_folder, f"sub-{subject_id}_ses-{session_id}_hemi-{hemi}_watershed_opt.gii")
rw.write_texture(stex.TextureND(darray=labels_array), gii_out)
print(f"[✓] Labels sauvegardés dans : {gii_out}")

# === Capture de screenshots ===
screenshots_path = os.path.join(output_folder, "screenshots")
print("Capture des vues du mesh segmenté...")
snap_mesh.capture_colored_mesh_snapshots(
    input_mesh=mesh_path,
    scalars=labels_array,
    output_path=screenshots_path,
    colormap="tab20"
)
print(f"[✓] Screenshots sauvegardés dans : {screenshots_path}")
