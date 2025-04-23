import os
import numpy as np
import glob
from joblib import Parallel, delayed
from numba import njit
from research.tools import rw
from research.tools import topology as slt
from research.tools import texture as stex
from research.tools.snapshot import snap_mesh

# === Paramètres globaux ===
data_root = "E:/rel3_dHCP_full"
dpf_folder = "E:/research_dpfstar/results_rel3_dhcp/dpfstar_smooth"
output_root = "E:/research_dpfstar/results_rel3_dhcp/watershed_labels_opt"
hemis = ["left", "right"]

# === Algorithme Watershed Optimisé ===
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

# === Fonction de traitement pour un sujet/session/hemisphère ===
def process_subject(subject_id, session_id, hemi):
    try:
        mesh_path = os.path.join(data_root, f"sub-{subject_id}/ses-{session_id}/anat",
                                 f"sub-{subject_id}_ses-{session_id}_hemi-{hemi}_wm.surf.gii")
        dpf_path = os.path.join(dpf_folder, f"hemi_{hemi}",
                                f"sub-{subject_id}_ses-{session_id}_hemi-{hemi}_dpfstar_smooth.gii")

        if not os.path.exists(mesh_path) or not os.path.exists(dpf_path):
            print(f"[SKIP] Fichiers manquants pour {subject_id} {session_id} {hemi}")
            return

        print(f"[INFO] Traitement : {subject_id} {session_id} {hemi}")
        mesh = rw.load_mesh(mesh_path)
        depth = rw.read_gii_file(dpf_path).flatten()

        adj_matrix = slt.adjacency_matrix(mesh)
        nb_vertices = mesh.vertices.shape[0]
        neighbors = [adj_matrix.indices[adj_matrix.indptr[i]:adj_matrix.indptr[i+1]] for i in range(nb_vertices)]

        # Watershed
        labels_array = run_watershed(depth, neighbors)
        n_regions = np.max(labels_array) + 1
        print(f"    → {n_regions} régions détectées.")

        # Sauvegarde
        output_folder = os.path.join(output_root, f"hemi_{hemi}")
        os.makedirs(output_folder, exist_ok=True)
        label_out = os.path.join(output_folder, f"sub-{subject_id}_ses-{session_id}_hemi-{hemi}_watershed_opt.gii")
        rw.write_texture(stex.TextureND(darray=labels_array), label_out)

        # Screenshots
        snap_mesh.capture_colored_mesh_snapshots(
            input_mesh=mesh_path,
            scalars=labels_array,
            output_path=os.path.join(output_folder, "screenshots", f"{subject_id}_{session_id}_{hemi}"),
            colormap="tab20"
        )

    except Exception as e:
        print(f"[ERREUR] {subject_id} {session_id} {hemi} → {e}")

# === Extraction automatique de tous les sujets/session ===
def get_subject_sessions():
    subject_paths = glob.glob(os.path.join(data_root, "sub-*"))
    subject_sessions = []
    for spath in subject_paths:
        subject_id = os.path.basename(spath).split("-")[1]
        sessions = glob.glob(os.path.join(spath, "ses-*"))
        for s in sessions:
            session_id = os.path.basename(s).split("-")[1]
            for hemi in hemis:
                subject_sessions.append((subject_id, session_id, hemi))
    return subject_sessions

# === MAIN ===
if __name__ == "__main__":
    all_combinations = get_subject_sessions()
    print(f"[INFO] Nombre total de combinaisons : {len(all_combinations)}")

    Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(process_subject)(subject, session, hemi)
        for subject, session, hemi in all_combinations
    )

    print("[✓] Traitement terminé.")
