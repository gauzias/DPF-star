import os
import pandas as pd
import numpy as np
import nibabel as nib
from glob import glob
from app import compute_dpfstar
from research.tools import rw

# Variables globales
path_rel3_dhcp = "E:/rel3_dHCP_full"
tsv_file = os.path.join("E:/rel3_dHCP", 'desc-drawem32_dseg.tsv')
info_df = pd.read_csv("E:/rel3_dHCP/info_rel3.csv")
dir_result = "E:/research_dpfstar/results_rel3_dhcp"

def extract_roi_data(tsv_file, gii_file):
    roi_df = pd.read_csv(tsv_file, sep='\t')
    roi_data = dict(zip(roi_df['index'], roi_df['name']))
    gii_data = rw.read_gii_file(gii_file)
    return gii_data, roi_data

def process_subject(subject_path, info_df, hemi, abr_hemi, save_csv_folder):
    subject_id = os.path.basename(subject_path).split('-')[1]
    sessions = glob(os.path.join(subject_path, 'ses-*'))
    all_session_data = []
    
    for session_path in sessions:
        session_id = os.path.basename(session_path).split('-')[1]
        anat_path = os.path.join(session_path, 'anat')
        if not os.path.exists(anat_path):
            continue

        nii_file = glob(os.path.join(anat_path, f'sub-{subject_id}_ses-{session_id}_hemi-{hemi}_desc-drawem_dseg.label.gii'))
        if not os.path.exists(tsv_file) or not nii_file:
            continue

        nii_file = nii_file[0]
        name_subject = f'sub-{subject_id}_ses-{session_id}_hemi-{hemi}_wm.surf'
        
        path_curvature = os.path.join(dir_result, "dpfstar", f'hemi_{hemi}', name_subject, "Kmean.gii")
        path_dpfstar = os.path.join(dir_result, "dpfstar", f'hemi_{hemi}', name_subject, "dpfstar.gii")
        if not os.path.exists(path_dpfstar):
            print('erreur', name_subject)
            continue

        dpfstar_values = rw.read_gii_file(path_dpfstar)
        curvature_values = rw.read_gii_file(path_curvature)

        subject_str = f"sub-{subject_id}"
        session_str = str(int(session_id))
        match = info_df[(info_df['suj'] == subject_str) & (info_df['session_id'].astype(str) == session_str)]

        scan_age = match['scan_age'].values[0] if not match.empty else np.nan
        volume = match['volume'].values[0] if not match.empty else np.nan
        GI = match['GI'].values[0] if not match.empty else np.nan
        surface = match['surface'].values[0] if not match.empty else np.nan

        roi_mask, roi_labels = extract_roi_data(tsv_file, nii_file)
        roi_labels = {label: name for label, name in roi_labels.items() if name.startswith(abr_hemi)}

        combined_rois = {
            "Temporal_lobe": [
                "Anterior_temporal_lobe_medial",
                "Anterior_temporal_lobe_lateral",
                "Superior_temporal_gyrus_middle",
                "Medial_and_inferior_temporal_gyri_anterior",
                "Medial_and_inferior_temporal_gyri_posterior"
            ]
        }

        # ROI individuelles
        for label, roi_name in roi_labels.items():
            roi_mask_indices = np.where(roi_mask == label)
            if len(roi_mask_indices[0]) == 0:
                continue

            roi_dpfstar = dpfstar_values[roi_mask_indices]
            roi_curvature = curvature_values[roi_mask_indices]
            negative_curvature_mask = roi_curvature < 0
            roi_dpfstar_neg_curv = roi_dpfstar[negative_curvature_mask]

            all_session_data.append({
                'Subject': subject_id,
                'Session': session_id,
                'hemi': hemi,
                'ROI': roi_name.split(abr_hemi)[1],
                'Label': label,
                'DPFstar_Mean_abs': np.mean(np.abs(roi_dpfstar)),
                'DPFstar_Mean': np.mean(roi_dpfstar),
                'DPFstar_Median': np.median(roi_dpfstar),
                'DPFstar_Min': np.min(roi_dpfstar),
                'DPFstar_NegCurv_Mean': np.mean(roi_dpfstar_neg_curv) if roi_dpfstar_neg_curv.size > 0 else np.nan,
                'DPFstar_NegCurv_Median': np.median(roi_dpfstar_neg_curv) if roi_dpfstar_neg_curv.size > 0 else np.nan,
                'DPFstar_NegCurv_Min': np.min(roi_dpfstar_neg_curv) if roi_dpfstar_neg_curv.size > 0 else np.nan,
                'scan_age': scan_age,
                'volume': volume,
                'GI': GI,
                'surface': surface
            })

        # ROI combinées
        for roi_name, components in combined_rois.items():
            indices = []
            for label, name in roi_labels.items():
                clean_name = name.split(abr_hemi)[1]
                if clean_name in components:
                    indices.append(np.where(roi_mask == label)[0])

            if not indices:
                continue

            combined_indices = np.concatenate(indices)
            roi_dpfstar = dpfstar_values[combined_indices]
            roi_curvature = curvature_values[combined_indices]
            negative_curvature_mask = roi_curvature < 0
            roi_dpfstar_neg_curv = roi_dpfstar[negative_curvature_mask]

            all_session_data.append({
                'Subject': subject_id,
                'Session': session_id,
                'hemi': hemi,
                'ROI': roi_name,
                'Label': -1,
                'DPFstar_Mean_abs': np.mean(np.abs(roi_dpfstar)),
                'DPFstar_Mean': np.mean(roi_dpfstar),
                'DPFstar_Median': np.median(roi_dpfstar),
                'DPFstar_Min': np.min(roi_dpfstar),
                'DPFstar_NegCurv_Mean': np.mean(roi_dpfstar_neg_curv) if roi_dpfstar_neg_curv.size > 0 else np.nan,
                'DPFstar_NegCurv_Median': np.median(roi_dpfstar_neg_curv) if roi_dpfstar_neg_curv.size > 0 else np.nan,
                'DPFstar_NegCurv_Min': np.min(roi_dpfstar_neg_curv) if roi_dpfstar_neg_curv.size > 0 else np.nan,
                'scan_age': scan_age,
                'volume': volume,
                'GI': GI,
                'surface': surface
            })

    return all_session_data

def main(data_root):
    hemispheres = [("right", "R_"), ("left", "L_")]

    subjects = glob(os.path.join(data_root, 'sub-*'))

    for hemi, abr_hemi in hemispheres:
        print(f"\n### Traitement hémisphère: {hemi} ###")
        save_csv_folder = os.path.join(dir_result, "cortical_metrics_dpfstar", f'hemi_{hemi}')
        os.makedirs(save_csv_folder, exist_ok=True)

        for subject in subjects:
            print(f"> Sujet : {subject}")
            subject_data = process_subject(subject, info_df, hemi, abr_hemi, save_csv_folder)

            df = pd.DataFrame(subject_data)
            if df.empty:
                print(f"Aucune donnée pour {subject}, hémisphère {hemi} — on saute.")
                continue

            for (subject_id, session_id), group_df in df.groupby(['Subject', 'Session']):
                output_file = os.path.join(
                    save_csv_folder,
                    f'sub-{subject_id}_ses-{session_id}_cortical_metrics.csv'
                )
                group_df.to_csv(output_file, index=False)
                print(f"Fichier généré : {output_file}")

if __name__ == "__main__":
    data_root = "E:/rel3_dHCP_full"
    main(data_root)
