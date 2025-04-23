import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from research.tools import rw

# === Paramètres ===
data_root = "E:/rel3_dHCP_full"
info_csv = "E:/rel3_dHCP/info_rel3.csv"
roi_tsv = "E:/rel3_dHCP/desc-drawem32_dseg.tsv"
deepest_root = "E:/research_dpfstar/results_rel3_dhcp/deepest_points"
dpf_smooth_root = "E:/research_dpfstar/results_rel3_dhcp/dpfstar_smooth"
save_csv = "E:/research_dpfstar/results_rel3_dhcp/deepest_summary_from_textures.csv"

# === Chargement des infos sujet et ROI
info_df = pd.read_csv(info_csv)
roi_df = pd.read_csv(roi_tsv, sep="\t")
roi_map = dict(zip(roi_df['index'], roi_df['name']))

hemis = ['left', 'right']
abr_hemi_map = {'left': 'L_', 'right': 'R_'}

results = []
subjects = sorted([s for s in os.listdir(data_root) if s.startswith("sub-")])

for subject in tqdm(subjects):
    subject_id = subject.split('-')[1]
    sessions = sorted(os.listdir(os.path.join(data_root, subject)))

    for session in sessions:
        session_id = session.split('-')[1]

        for hemi in hemis:
            abr = abr_hemi_map[hemi]

            try:
                # === Paths
                anat_dir = os.path.join(data_root, subject, session, "anat")
                label_path = os.path.join(anat_dir, f"sub-{subject_id}_ses-{session_id}_hemi-{hemi}_desc-drawem_dseg.label.gii")
                dpf_path = os.path.join(dpf_smooth_root, f"hemi_{hemi}", f"sub-{subject_id}_ses-{session_id}_hemi-{hemi}_dpfstar_smooth.gii")
                deepest_path = os.path.join(deepest_root, f"hemi_{hemi}", f"sub-{subject_id}_ses-{session_id}_hemi-{hemi}_deepest_point.gii")

                if not (os.path.exists(label_path) and os.path.exists(dpf_path) and os.path.exists(deepest_path)):
                    continue

                drawem_labels = rw.read_gii_file(label_path).astype(int)
                dpf = rw.read_gii_file(dpf_path).flatten()
                deepest = rw.read_gii_file(deepest_path).flatten()

                if not (len(drawem_labels) == len(dpf) == len(deepest)):
                    print(f"[!] Taille incohérente: {subject_id} {session_id} {hemi}")
                    continue

                indices = np.arange(len(dpf))
                roi_ids = [k for k, v in roi_map.items() if v.startswith(abr)]
                roi_subset = {k: roi_map[k] for k in roi_ids}

                subject_str = f"sub-{subject_id}"
                session_str = str(int(session_id))
                match = info_df[(info_df["suj"] == subject_str) & (info_df["session_id"].astype(str) == session_str)]
                scan_age = pd.to_numeric(match["scan_age"].values[0], errors="coerce") if not match.empty else np.nan

                for roi_idx, roi_name in roi_subset.items():
                    roi_mask = drawem_labels == roi_idx
                    roi_indices = indices[roi_mask]

                    if roi_indices.size == 0:
                        continue

                    deepest_indices = roi_indices[deepest[roi_indices] == 1]
                    n_deepest = len(deepest_indices)
                    mean_depth = float(np.mean(dpf[deepest_indices])) if n_deepest > 0 else 0.0

                    results.append({
                        "Subject": subject_id,
                        "Session": session_id,
                        "Hemisphere": hemi,
                        "ROI": roi_name,
                        "Age": scan_age,
                        "DeepestPoints": n_deepest,
                        "MeanDepth": mean_depth
                    })

            except Exception as e:
                print(f"[!] Erreur {subject_id} {session_id} {hemi} : {e}")

# === Export CSV ===
df = pd.DataFrame(results)
df.to_csv(save_csv, index=False)
print(f"[✓] Résultats sauvegardés dans : {save_csv}")
