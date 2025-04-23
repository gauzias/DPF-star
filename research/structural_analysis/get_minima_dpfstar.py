import numpy as np
import pandas as pd
import os
import glob
import traceback
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import Parallel, delayed

from research.tools import rw
from research.tools import topology as slt
from research.tools import texture as stex
from research.tools import differential_geometry as sdg

# === Globales ===
path_rel3_dhcp = "E:/rel3_dHCP_full"
tsv_file = os.path.join("E:/rel3_dHCP", 'desc-drawem32_dseg.tsv')
info_df = pd.read_csv("E:/rel3_dHCP/info_rel3.csv")
dir_result = "E:/research_dpfstar/results_rel3_dhcp"
save_csv_folder = os.path.join(dir_result, "minima_count")
save_smooth_folder = os.path.join(dir_result, "dpfstar_smooth")
os.makedirs(save_csv_folder, exist_ok=True)
os.makedirs(save_smooth_folder, exist_ok=True)

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

def extract_roi_data(tsv_file, gii_file):
    roi_df = pd.read_csv(tsv_file, sep='\t')
    roi_data = dict(zip(roi_df['index'], roi_df['name']))
    gii_data = rw.read_gii_file(gii_file)
    return gii_data, roi_data

def process_subject(subject_path):
    subject_id = os.path.basename(subject_path).split('-')[1]
    sessions = glob.glob(os.path.join(subject_path, 'ses-*'))
    all_data = []

    try:
        for hemi in hemis:
            abr_hemi = abr_hemi_map[hemi]

            for session_path in sessions:
                session_id = os.path.basename(session_path).split('-')[1]
                anat_path = os.path.join(session_path, 'anat')
                if not os.path.exists(anat_path):
                    continue

                surface_file = glob.glob(os.path.join(anat_path, f'sub-{subject_id}_ses-{session_id}_hemi-{hemi}_wm.surf.gii'))
                label_file = glob.glob(os.path.join(anat_path, f'sub-{subject_id}_ses-{session_id}_hemi-{hemi}_desc-drawem_dseg.label.gii'))
                path_depth = os.path.join(dir_result, "dpfstar", f'hemi_{hemi}', f'sub-{subject_id}_ses-{session_id}_hemi-{hemi}_wm.surf', "dpfstar.gii")

                if not (surface_file and os.path.exists(path_depth) and label_file):
                    continue

                mesh = rw.load_mesh(surface_file[0])
                depth = rw.read_gii_file(path_depth)

                depth_smoothed = sdg.gaussian_smoothing(mesh, depth, fwhm=10)

                smooth_hemi_dir = os.path.join(save_smooth_folder, f"hemi_{hemi}")
                os.makedirs(smooth_hemi_dir, exist_ok=True)
                smooth_file = f"sub-{subject_id}_ses-{session_id}_hemi-{hemi}_dpfstar_smooth.gii"
                rw.write_texture(stex.TextureND(darray=depth_smoothed), os.path.join(smooth_hemi_dir, smooth_file))

                roi_mask, roi_labels = extract_roi_data(tsv_file, label_file[0])
                minima = get_minima(mesh, depth_smoothed)

                subject_str = f"sub-{subject_id}"
                session_str = str(int(session_id))
                match = info_df[(info_df['suj'] == subject_str) & (info_df['session_id'].astype(str) == session_str)]
                scan_age = pd.to_numeric(match['scan_age'].values[0], errors="coerce") if not match.empty else np.nan

                for label, roi_name in roi_labels.items():
                    if not roi_name.startswith(abr_hemi):
                        continue
                    indices = np.where(roi_mask == label)[0]
                    n_minima = np.sum(minima[indices]) if indices.size > 0 else 0

                    all_data.append({
                        'Subject': subject_id,
                        'Session': session_id,
                        'Hemisphere': hemi,
                        'ROI': roi_name,
                        'Age': scan_age,
                        'MinimaCount': n_minima
                    })

    except Exception as e:
        print(f"[!] Erreur avec {subject_id} : {e}")
        traceback.print_exc()
    return all_data


def generate_plots(df_summary):
    sns.set_theme(style="whitegrid")
    df_summary["Age"] = pd.to_numeric(df_summary["Age"], errors="coerce")
    df_summary = df_summary.dropna(subset=["Age"])

    rois = sorted(df_summary['ROI'].unique())
    cols = 4
    rows = (len(rois) + cols - 1) // cols

    # === Figure 1 : regression ===
    fig1, axes1 = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), constrained_layout=True)
    axes1 = axes1.flatten()

    for i, roi in enumerate(rois):
        ax = axes1[i]
        for hemi, color in zip(['left', 'right'], ['blue', 'red']):
            data = df_summary[(df_summary['ROI'] == roi) & (df_summary['Hemisphere'] == hemi)]
            if not data.empty:
                sns.regplot(data=data, x="Age", y="MinimaCount", scatter_kws={'s': 20}, line_kws={'color': color}, ax=ax, label=hemi.capitalize())
        ax.set_title(roi, fontsize=9)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Scan Age")
        ax.set_ylabel("Minima Count")
        ax.legend()

    for j in range(i+1, len(axes1)):
        fig1.delaxes(axes1[j])

    fig1_path = os.path.join(save_csv_folder, "figure_minima_regression.png")
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    print(f"[Figure 1] Sauvegardée : {fig1_path}")

    # === Figure 2 : scatter brut ===
    fig2, axes2 = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), constrained_layout=True)
    axes2 = axes2.flatten()

    for i, roi in enumerate(rois):
        ax = axes2[i]
        for hemi, color in zip(['left', 'right'], ['blue', 'red']):
            data = df_summary[(df_summary['ROI'] == roi) & (df_summary['Hemisphere'] == hemi)]
            if not data.empty:
                ax.scatter(data['Age'], data['MinimaCount'], color=color, alpha=0.6, label=hemi.capitalize(), s=20)
        ax.set_title(roi, fontsize=9)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Scan Age")
        ax.set_ylabel("Minima Count")
        ax.legend()

    for j in range(i+1, len(axes2)):
        fig2.delaxes(axes2[j])

    fig2_path = os.path.join(save_csv_folder, "figure_minima_scatter.png")
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    print(f"[Figure 2] Sauvegardée : {fig2_path}")


def main(data_root):
    subjects = glob.glob(os.path.join(data_root, 'sub-*'))
    print(f"[INFO] Nombre de sujets à traiter : {len(subjects)}")

    # Traitement parallèle
    results = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(process_subject)(s) for s in subjects)

    # Flatten
    all_results = [item for sublist in results for item in sublist if sublist]

    # Résumé
    df_summary = pd.DataFrame(all_results)
    df_summary["Age"] = pd.to_numeric(df_summary["Age"], errors="coerce")
    df_summary = df_summary.dropna(subset=["Age"])

    # Sauvegarde CSV
    summary_csv = os.path.join(save_csv_folder, "minima_summary.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"[CSV] Données sauvegardées : {summary_csv}")

    # Figures
    generate_plots(df_summary)


if __name__ == "__main__":
    main("E:/rel3_dHCP_full")
