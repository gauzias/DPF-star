import os
import pandas as pd
import numpy as np
import nibabel as nib
import datetime
import platform
import socket
import psutil
from glob import glob
from app import compute_dpfstar
from research.tools import rw
from fpdf import FPDF
import unicodedata
import traceback
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def remove_accents(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

def extract_roi_data(tsv_file, gii_file):
    roi_df = pd.read_csv(tsv_file, sep='\t')
    roi_data = dict(zip(roi_df['index'], roi_df['name']))
    gii_data = rw.read_gii_file(gii_file)
    return gii_data, roi_data

def generate_execution_report(start_time, end_time, total_subjects, pdf_path):
    duration = round((end_time - start_time).total_seconds(), 2)
    ram_used = round((psutil.virtual_memory().total - psutil.virtual_memory().available) / (1024**3), 2)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, remove_accents("Resume d'Execution - DPFStar Pipeline"), ln=True, align="C")

    pdf.set_font("Arial", '', 12)
    lines = [
        f"Date : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Machine : {socket.gethostname()}",
        f"Systeme : {platform.system()} {platform.version()}",
        f"Processeur : {platform.processor()}",
        f"CPU Coeurs (Physiques) : {psutil.cpu_count(logical=False)}",
        f"CPU Coeurs (Logiques) : {psutil.cpu_count(logical=True)}",
        f"RAM Totale : {round(psutil.virtual_memory().total / (1024**3), 2)} Go",
        f"RAM Utilisee durant l'execution : {ram_used} Go",
        f"Temps d'execution total : {duration} secondes",
        f"Sujets traites : {total_subjects}"
    ]

    for line in lines:
        pdf.cell(0, 10, remove_accents(line), ln=True)

    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    pdf.output(pdf_path)

def process_subject(subject_path, info_df_path, save_csv_folder):
    import pandas as pd
    import numpy as np
    import traceback

    log_file = os.path.join(save_csv_folder, "execution_log.txt")
    error_log_file = os.path.join(save_csv_folder, "error_log.txt")
    info_df = pd.read_csv(info_df_path)

    subject_id = os.path.basename(subject_path).split('-')[1]
    sessions = glob(os.path.join(subject_path, 'ses-*'))
    all_session_data = []

    for session_path in sessions:
        session_id = os.path.basename(session_path).split('-')[1]
        try:
            anat_path = os.path.join(session_path, 'anat')
            if not os.path.exists(anat_path):
                continue

            surface_file = glob(os.path.join(anat_path, f'sub-{subject_id}_ses-{session_id}_hemi-left_wm.surf.gii'))
            tsv_file = os.path.join("D:/Callisto/data/rel3_dhcp", 'desc-drawem32_dseg.tsv')
            nii_file = glob(os.path.join(anat_path, f'sub-{subject_id}_ses-{session_id}_hemi-left_desc-drawem_dseg.label.gii'))

            if not surface_file or not os.path.exists(tsv_file) or not nii_file:
                continue

            surface_file = surface_file[0]
            nii_file = nii_file[0]

            compute_dpfstar.main(surface_file, display=False)
            with open(log_file, 'a') as log:
                log.write(f"{subject_id}_{session_id} : DPFStar et courbure calcules.\n")

            name_subject = f'sub-{subject_id}_ses-{session_id}_hemi-left_wm.surf'
            dir_dpfstar = "D:/Callisto/data/results_rel3_dhcp"
            path_curvature = os.path.join(dir_dpfstar, name_subject, "Kmean.gii")
            path_dpfstar = os.path.join(dir_dpfstar, name_subject, "dpfstar.gii")

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

            rows = []
            for label, roi_name in roi_labels.items():
                roi_mask_indices = np.where(roi_mask == label)
                if len(roi_mask_indices[0]) == 0:
                    continue

                roi_dpfstar = dpfstar_values[roi_mask_indices]
                roi_curvature = curvature_values[roi_mask_indices]
                negative_curvature_mask = roi_curvature < 0
                roi_dpfstar_neg_curv = roi_dpfstar[negative_curvature_mask]

                rows.append({
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

            if rows:
                df = pd.DataFrame(rows)
                output_file = os.path.join(save_csv_folder, f'sub-{subject_id}_ses-{session_id}_cortical_metrics.csv')
                df.to_csv(output_file, index=False)

        except Exception as e:
            with open(error_log_file, 'a') as errlog:
                errlog.write(f"Erreur avec {subject_id}_{session_id}: {str(e)}\n")
                errlog.write(traceback.format_exc() + "\n")

def main(data_root):
    start_time = datetime.datetime.now()
    info_df_path = "D:/Callisto/data/rel3_dhcp/info_rel3.csv"
    subjects = glob(os.path.join(data_root, 'sub-*'))
    save_csv_folder = os.path.join("D:/Callisto/data/results_rel3_dhcp", "cortical_metrics")
    os.makedirs(save_csv_folder, exist_ok=True)

    with open(os.path.join(save_csv_folder, "execution_log.txt"), 'w') as log:
        log.write("Execution Log - DPFStar Pipeline\n\n")
    with open(os.path.join(save_csv_folder, "error_log.txt"), 'w') as err:
        err.write("Error Log - DPFStar Pipeline\n\n")

    worker = partial(process_subject, info_df_path=info_df_path, save_csv_folder=save_csv_folder)

    with ProcessPoolExecutor() as executor:
        executor.map(worker, subjects)

    end_time = datetime.datetime.now()
    pdf_path = os.path.join(save_csv_folder, "summary_execution_report.pdf")
    generate_execution_report(start_time, end_time, total_subjects=len(subjects), pdf_path=pdf_path)

if __name__ == "__main__":
    data_root = "E:/rel3_dHCP"
    main(data_root)
