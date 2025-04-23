import os
import numpy as np
from tqdm import tqdm
from research.tools import rw
from research.tools import texture as stex

# === Chemins ===
base_dir = "E:/research_dpfstar/results_rel3_dhcp"
watershed_dir = os.path.join(base_dir, "watershed_labels_opt")
dpf_dir = os.path.join(base_dir, "dpfstar_smooth")
output_dir = os.path.join(base_dir, "deepest_points")
os.makedirs(output_dir, exist_ok=True)

hemis = ['left', 'right']

def compute_deepest_points(depth, labels):
    """
    Pour chaque label de région, retourne un tableau binaire contenant
    uniquement le point avec la valeur de profondeur minimale.
    """
    output = np.zeros_like(depth, dtype=np.uint8)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]

    for lbl in unique_labels:
        indices = np.where(labels == lbl)[0]
        if indices.size == 0:
            continue
        region_depths = depth[indices]
        min_val = np.min(region_depths)
        min_indices = indices[region_depths == min_val]
        selected = min_indices[0]  # on prend seulement un point
        output[selected] = 1
    return output


for hemi in hemis:
    hemi_dir = os.path.join(watershed_dir, f"hemi_{hemi}")
    output_hemi_dir = os.path.join(output_dir, f"hemi_{hemi}")
    os.makedirs(output_hemi_dir, exist_ok=True)

    for file in tqdm(os.listdir(hemi_dir), desc=f"[{hemi.upper()}]"):
        if not file.endswith("_watershed_opt.gii"):
            continue

        # Identifiants
        basename = file.replace("_watershed_opt.gii", "")
        subject_parts = basename.split("_")
        subject_id = subject_parts[0].replace("sub-", "")
        session_id = subject_parts[1].replace("ses-", "")
        dpf_path = os.path.join(dpf_dir, f"hemi_{hemi}", f"{basename.replace('_watershed', '')}_dpfstar_smooth.gii")
        watershed_path = os.path.join(hemi_dir, file)
        output_path = os.path.join(output_hemi_dir, f"{basename}_deepest_point.gii")

        if not os.path.exists(dpf_path):
            print(f"[!] Fichier manquant : {dpf_path}")
            continue

        try:
            depth = rw.read_gii_file(dpf_path).flatten()
            labels = rw.read_gii_file(watershed_path).flatten().astype(int)
            bin_texture = compute_deepest_points(depth, labels)
            rw.write_texture(stex.TextureND(darray=bin_texture), output_path)
        except Exception as e:
            print(f"[ERROR] {basename} : {e}")
