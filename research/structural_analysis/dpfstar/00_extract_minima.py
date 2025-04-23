import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

from research.tools import rw
from research.tools import topology as slt
from research.tools import texture as stex

# === Paramètres ===
data_root = "E:/rel3_dHCP_full"
dir_smooth = "E:/research_dpfstar/results_rel3_dhcp/dpfstar_smooth"
output_dir = "E:/research_dpfstar/results_rel3_dhcp/minima_textures"
os.makedirs(output_dir, exist_ok=True)

hemis = ['left', 'right']
abr_hemi_map = {'left': 'L_', 'right': 'R_'}

def get_minima(mesh, texture):
    nbv = texture.shape[0]
    adj_matrix = slt.adjacency_matrix(mesh)
    minima = np.zeros((nbv,))
    for idx, tex_i in enumerate(texture):
        neigh = adj_matrix.indices[adj_matrix.indptr[idx]:adj_matrix.indptr[idx + 1]]
        neigh_tex = texture[neigh]
        if tex_i <= neigh_tex.min():
            minima[idx] = 1
    return minima

def process_subject(subject_path):
    subject_id = os.path.basename(subject_path).split('-')[1]
    sessions = glob.glob(os.path.join(subject_path, 'ses-*'))

    for hemi in hemis:
        for session_path in sessions:
            session_id = os.path.basename(session_path).split('-')[1]
            anat_path = os.path.join(session_path, 'anat')
            if not os.path.exists(anat_path):
                continue

            # === Paths ===
            surf_file = glob.glob(os.path.join(anat_path, f'sub-{subject_id}_ses-{session_id}_hemi-{hemi}_wm.surf.gii'))
            smooth_file = os.path.join(
                dir_smooth, f"hemi_{hemi}", f"sub-{subject_id}_ses-{session_id}_hemi-{hemi}_dpfstar_smooth.gii"
            )
            output_file = os.path.join(
                output_dir, f"sub-{subject_id}_ses-{session_id}_hemi-{hemi}_minima.gii"
            )

            if not surf_file or not os.path.exists(smooth_file):
                continue

            # === Chargement ===
            mesh = rw.load_mesh(surf_file[0])
            texture = rw.read_gii_file(smooth_file)

            # === Minima ===
            minima_mask = get_minima(mesh, texture)

            # === Sauvegarde ===
            tex = stex.TextureND(darray=minima_mask.astype(np.float32))
            rw.write_texture(tex, output_file)

if __name__ == "__main__":
    subjects = glob.glob(os.path.join(data_root, 'sub-*'))
    print(f"[INFO] Traitement de {len(subjects)} sujets...")

    for sub in tqdm(subjects):
        process_subject(sub)

    print("Sauvegarde des textures des minima terminée.")
