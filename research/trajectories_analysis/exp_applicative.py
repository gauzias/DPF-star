import os
import pandas as pd
import numpy as np
import nibabel as nib
from glob import glob
from app import compute_dpfstar
from research.tools import rw

def extract_roi_data(tsv_file, gii_file):
    roi_df = pd.read_csv(tsv_file, sep='\t')
    roi_data = dict(zip(roi_df['index'], roi_df['name']))
    gii_data = rw.read_gii_file(gii_file)
    return gii_data, roi_data

def process_subject(subject_path, info_df):
    subject_id = os.path.basename(subject_path).split('-')[1]
    sessions = glob(os.path.join(subject_path, 'ses-*'))
    all_session_data = []

    for session_path in sessions:
        session_id = os.path.basename(session_path).split('-')[1]
        anat_path = os.path.join(session_path, 'anat')
        if not os.path.exists(anat_path):
            continue

        # Fichiers requis
        surface_file = glob(os.path.join(anat_path, f'sub-{subject_id}_ses-{session_id}_hemi-left_wm.surf.gii'))
        tsv_file = os.path.join("D:/Callisto/data/rel3_dhcp", 'desc-drawem32_dseg.tsv')
        nii_file = glob(os.path.join(anat_path, f'sub-{subject_id}_ses-{session_id}_hemi-left_desc-drawem_dseg.label.gii'))

        if not surface_file or not os.path.exists(tsv_file) or not nii_file:
            continue

        surface_file = surface_file[0]
        nii_file = nii_file[0]

        compute_dpfstar.main(surface_file, display=False)  
        print("compute dpstar and curvature done")

        name_subject = f'sub-{subject_id}_ses-{session_id}_hemi-left_wm.surf'
        dir_dpfstar = "D:/Callisto/data/results_rel3_dhcp"
        path_curvature = os.path.join(dir_dpfstar, name_subject, "Kmean.gii")
        path_dpfstar = os.path.join(dir_dpfstar, name_subject, "dpfstar.gii")

        dpfstar_values = rw.read_gii_file(path_dpfstar)
        curvature_values = rw.read_gii_file(path_curvature)

        # Infos complémentaires depuis info_rel3.csv
        subject_str = f"sub-{subject_id}"
        session_str = str(int(session_id))
        match = info_df[(info_df['suj'] == subject_str) & (info_df['session_id'].astype(str) == session_str)]

        scan_age = match['scan_age'].values[0] if not match.empty else np.nan
        volume = match['volume'].values[0] if not match.empty else np.nan
        GI = match['GI'].values[0] if not match.empty else np.nan
        surface = match['surface'].values[0] if not match.empty else np.nan

        # Extraction ROI
        roi_mask, roi_labels = extract_roi_data(tsv_file, nii_file)

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
                'ROI': roi_name,
                'Label': label,
                'DPFStar_Mean': np.mean(roi_dpfstar),
                'DPFStar_Median': np.median(roi_dpfstar),
                'DPFStar_Min': np.min(roi_dpfstar),
                'DPFStar_NegCurv_Mean': np.mean(roi_dpfstar_neg_curv) if roi_dpfstar_neg_curv.size > 0 else np.nan,
                'DPFStar_NegCurv_Median': np.median(roi_dpfstar_neg_curv) if roi_dpfstar_neg_curv.size > 0 else np.nan,
                'DPFStar_NegCurv_Min': np.min(roi_dpfstar_neg_curv) if roi_dpfstar_neg_curv.size > 0 else np.nan,
                'scan_age': scan_age,
                'volume': volume,
                'GI': GI,
                'surface': surface
            })

    return all_session_data

def main(data_root):
    info_df = pd.read_csv("D:/Callisto/data/rel3_dhcp/info_rel3.csv")
    subjects = glob(os.path.join(data_root, 'sub-*'))
    save_csv_folder = os.path.join("D:/Callisto/data/results_rel3_dhcp", "cortical_metrics")
    os.makedirs(save_csv_folder, exist_ok=True)

    for subject in subjects:
        print(subject)
        subject_data = process_subject(subject, info_df)
        df = pd.DataFrame(subject_data)

        if df.empty:
            print(f"Aucune donnée pour {subject}, on saute.")
            continue

        for (subject_id, session_id), group_df in df.groupby(['Subject', 'Session']):
            output_file = os.path.join(
                save_csv_folder,
                f'sub-{subject_id}_ses-{session_id}_cortical_metrics.csv'
            )
            group_df.to_csv(output_file, index=False)
            print(f"Fichier généré : {output_file}")

if __name__ == "__main__":
    #data_root = "D:/Callisto/data/rel3_dhcp"
    data_root = "E:/rel3_dHCP"
    main(data_root)
