import os
import pandas as pd
import numpy as np
import nibabel as nib
from glob import glob
from app import compute_dpfstar
from research.tools import rw


# variables globales
path_rel3_dhcp = "E:/rel3_dHCP_full"
tsv_file = os.path.join("E:/rel3_dHCP", 'desc-drawem32_dseg.tsv')
#hemi = "right"
#abr_hemi = "R_"
hemi = "left"
abr_hemi = "L_"
dir_result = "E:/research_dpfstar/results_rel3_dhcp"
info_df = pd.read_csv("E:/rel3_dHCP/info_rel3.csv")
save_csv_folder = os.path.join(dir_result, "cortical_metrics_sulc", f'hemi_{hemi}')

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
        nii_file = glob(os.path.join(anat_path, f'sub-{subject_id}_ses-{session_id}_hemi-{hemi}_desc-drawem_dseg.label.gii'))

        if not os.path.exists(tsv_file) or not nii_file:
            continue

        nii_file = nii_file[0]

        #compute_dpfstar.main(surface_file, display=False)  
        #print("compute sulc done")

        name_subject = f'sub-{subject_id}_ses-{session_id}_hemi-{hemi}_wm.surf'
        
        path_curvature = os.path.join(dir_result, "dpfstar", f'hemi_{hemi}', name_subject, "Kmean.gii")
        path_dpfstar = os.path.join(anat_path, f'sub-{subject_id}_ses-{session_id}_hemi-{hemi}_sulc.shape.gii') #sulc

        if not os.path.exists(path_dpfstar):
            print('erreur', name_subject)
            continue

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

        roi_labels = {label: name for label, name in roi_labels.items() if name.startswith(abr_hemi)}
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
                'hemi' : hemi,
                'ROI': roi_name.split(abr_hemi)[1],
                'Label': label,
                'Sulc_Mean_abs' : np.mean(np.abs(roi_dpfstar)),
                'Sulc_Min': np.min(roi_dpfstar),
                'Sulc_p5' : np.percentile(roi_dpfstar,5),
                'Sulc_p10' : np.percentile(roi_dpfstar,10),
                'Sulc_p15' : np.percentile(roi_dpfstar,15),
                'Sulc_p20' : np.percentile(roi_dpfstar,20),
                'Sulc_Q1' : np.percentile(roi_dpfstar,25),
                'Sulc_Mean': np.mean(roi_dpfstar),
                'Sulc_Median': np.median(roi_dpfstar),
                'Sulc_Q3' : np.percentile(roi_dpfstar, 75) ,
                'Sulc_Max': np.max(roi_dpfstar),
                'Sulc_NegCurv_Min': np.min(roi_dpfstar_neg_curv) if roi_dpfstar_neg_curv.size > 0 else np.nan,
                'Sulc_NegCurv_Q1': np.percentile(roi_dpfstar_neg_curv,25) if roi_dpfstar_neg_curv.size > 0 else np.nan,
                'Sulc_NegCurv_Mean': np.mean(roi_dpfstar_neg_curv) if roi_dpfstar_neg_curv.size > 0 else np.nan,
                'Sulc_NegCurv_Median': np.median(roi_dpfstar_neg_curv) if roi_dpfstar_neg_curv.size > 0 else np.nan,
                'Sulc_NegCurv_Q3': np.percentile(roi_dpfstar_neg_curv,75) if roi_dpfstar_neg_curv.size > 0 else np.nan,
                'Sulc_NegCurv_Max': np.max(roi_dpfstar_neg_curv) if roi_dpfstar_neg_curv.size > 0 else np.nan,
                'scan_age': scan_age,
                'volume': volume,
                'GI': GI,
                'surface': surface
            })

    return all_session_data

def main(data_root):
    subjects = glob(os.path.join(data_root, 'sub-*'))
    
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
    data_root = "E:/rel3_dHCP_full"
    main(data_root)
