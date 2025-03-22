import os
import pandas as pd
import numpy as np
import nibabel as nib
from glob import glob
from app import compute_dpfstar
from research.tools import rw

def extract_roi_data(tsv_file, gii_file):
    """Extrait les labels des ROI depuis le fichier TSV et applique les masques."""
    roi_df = pd.read_csv(tsv_file, sep='\t')
    roi_data = dict(zip(roi_df['index'], roi_df['name']))
    #nii_data = nib.load(nii_file).get_fdata()
    gii_data = rw.read_gii_file(gii_file)
    return gii_data, roi_data

def process_subject(subject_path):
    subject_id = os.path.basename(subject_path).split('-')[1]
    sessions = glob(os.path.join(subject_path, 'ses-*'))

    subject_data = []

    for session_path in sessions:
        session_id = os.path.basename(session_path).split('-')[1]
        anat_path = os.path.join(session_path, 'anat')

        if not os.path.exists(anat_path):
            continue

        # Récupérer les fichiers nécessaires
        surface_file = glob(os.path.join(anat_path, f'sub-{subject_id}_ses-{session_id}_hemi-left_wm.surf.gii'))
        tsv_file = os.path.join("D:/Callisto/data/rel3_dhcp", 'desc-drawem32_dseg.tsv')
        nii_file = glob(os.path.join(anat_path, f'sub-{subject_id}_ses-{session_id}_hemi-left_desc-drawem_dseg.label.gii'))

        if not surface_file or not os.path.exists(tsv_file) or not nii_file:
            continue

        surface_file = surface_file[0]
        nii_file = nii_file[0]

        # Calcul de la dpfstar et de la courbure
        dir_dpfstar = "D:/Callisto/data/results_rel3_dhcp"
        #compute_dpfstar.main(surface_file,  display=False)
        print("compute dpstar and curvature done")

        name_subject = f'sub-{subject_id}_ses-{session_id}_hemi-left_wm.surf'
        path_curvature = os.path.join(dir_dpfstar, name_subject, "Kmean.gii")
        path_dpfstar = os.path.join(dir_dpfstar, name_subject, "dpfstar.gii")

        dpfstar_values = rw.read_gii_file(path_dpfstar)
        curvature_values = rw.read_gii_file(path_curvature)

        # Extraction des ROI
        roi_mask, roi_labels = extract_roi_data(tsv_file, nii_file)

        # Calcul des statistiques
        for label, roi_name in roi_labels.items():
            roi_mask_indices = np.where(roi_mask == label)
            if len(roi_mask_indices[0]) == 0:
                continue

            roi_dpfstar = dpfstar_values[roi_mask_indices]
            roi_curvature = curvature_values[roi_mask_indices]
            negative_curvature_mask = roi_curvature < 0
            roi_dpfstar_neg_curv = roi_dpfstar[negative_curvature_mask]

            subject_data.append({
                'Subject': subject_id,
                'Session': session_id,
                'ROI': roi_name,
                'DPFStar_Mean': np.mean(roi_dpfstar),
                'DPFStar_Median': np.median(roi_dpfstar),
                'DPFStar_Max': np.max(roi_dpfstar),
                'DPFStar_NegCurv_Mean': np.mean(roi_dpfstar_neg_curv) if roi_dpfstar_neg_curv.size > 0 else np.nan,
                'DPFStar_NegCurv_Median': np.median(roi_dpfstar_neg_curv) if roi_dpfstar_neg_curv.size > 0 else np.nan,
                'DPFStar_NegCurv_Max': np.max(roi_dpfstar_neg_curv) if roi_dpfstar_neg_curv.size > 0 else np.nan,
            })

    return subject_data

def main(data_root):
    subjects = glob(os.path.join(data_root, 'sub-*'))
    all_data = []

    for subject in subjects:
        all_data.extend(process_subject(subject))

    df = pd.DataFrame(all_data)

    # Sauvegarde du tableau
    output_file = os.path.join(data_root, 'cortical_metrics.csv')
    df.to_csv(output_file, index=False)

    print(f"Tableau généré : {output_file}")
    return df

if __name__ == "__main__":
    #data_root = "E:/rel3_dHCP"
    data_root = "D:/Callisto/data/rel3_dhcp"
    df_result = main(data_root)