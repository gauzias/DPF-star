import os
import pandas as pd
import numpy as np
import nibabel as nib
from glob import glob
from app import compute_dpfstar
from research.tools import rw
from concurrent.futures import ProcessPoolExecutor
import traceback
import time
import platform
import psutil
import socket
from fpdf import FPDF

# Variables globales
path_rel3_dhcp = "E:/rel3_dhcp_full"
#hemi = "right"
hemi = "left"

def generate_report(start_time, end_time, total_subjects, output_path):
    duration = round((end_time - start_time), 2)
    ram_used = round((psutil.virtual_memory().total - psutil.virtual_memory().available) / (1024**3), 2)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Résumé d'Exécution - DPFStar", ln=True, align="C")

    pdf.set_font("Arial", '', 12)
    lines = [
        f"Date : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Machine : {socket.gethostname()}",
        f"Système : {platform.system()} {platform.version()}",
        f"Processeur : {platform.processor()}",
        f"CPU (physiques/logiques) : {psutil.cpu_count(logical=False)}/{psutil.cpu_count(logical=True)}",
        f"RAM totale : {round(psutil.virtual_memory().total / (1024**3), 2)} Go",
        f"RAM utilisée durant l'exécution : {ram_used} Go",
        f"Durée totale : {duration} secondes",
        f"Nombre de sujets traités : {total_subjects}"
    ]

    for line in lines:
        pdf.cell(0, 10, line, ln=True)

    pdf.output(output_path)

def process_subject(subject_path, log_file, error_log_file):
    subject_id = os.path.basename(subject_path).split('-')[1]
    sessions = glob(os.path.join(subject_path, 'ses-*'))

    try:
        for session_path in sessions:
            session_id = os.path.basename(session_path).split('-')[1]
            anat_path = os.path.join(session_path, 'anat')
            if not os.path.exists(anat_path):
                continue

            surface_file = glob(os.path.join(anat_path, f'sub-{subject_id}_ses-{session_id}_hemi-{hemi}_wm.surf.gii'))
            if not surface_file:
                continue

            surface_file = surface_file[0]
            compute_dpfstar.main(surface_file, display=False)

            with open(log_file, 'a') as log:
                log.write(f"{subject_id}_{session_id} : traitement OK.\n")
            print(f"{subject_id}_{session_id} : compute dpfstar done")

    except Exception as e:
        with open(error_log_file, 'a') as errlog:
            errlog.write(f"Erreur avec {subject_id}: {str(e)}\n")
            errlog.write(traceback.format_exc() + "\n")
        print(f"{subject_id}: erreur détectée")

def main(data_root):
    start_time = time.time()
    subjects = glob(os.path.join(data_root, 'sub-*'))
    print(subjects)

    # Préparer les logs
    log_file = os.path.join(data_root, "log_execution.txt")
    error_log_file = os.path.join(data_root, "log_erreurs.txt")
    with open(log_file, 'w') as f:
        f.write("Log des traitements DPFStar\n\n")
    with open(error_log_file, 'w') as f:
        f.write("Log des erreurs DPFStar\n\n")

    # Préparer fonction partielle pour le pool
    from functools import partial
    worker = partial(process_subject, log_file=log_file, error_log_file=error_log_file)

    # Traitement parallèle
    with ProcessPoolExecutor() as executor:
        executor.map(worker, subjects)

    end_time = time.time()
    report_path = os.path.join(data_root, "resume_execution_dpfstar.pdf")
    generate_report(start_time, end_time, len(subjects), report_path)
    print(f"\n✅ Rapport d'exécution généré : {report_path}")

if __name__ == "__main__":
    #data_root = "E:/rel3_dHCP_full"
    data_root = "E:/rel3_dhcp_full"
    main(data_root)
