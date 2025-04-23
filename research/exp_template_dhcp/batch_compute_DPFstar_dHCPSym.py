import os
import time
import platform
import socket
import psutil
import traceback
from glob import glob
from fpdf import FPDF
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from app import compute_dpfstar

# === CONFIGURATION ===
path_dataset = r"E:\dhcpSym_template"
output_root = r"E:/research_dpfstar/result_dhcpSym"
hemi_id = ["left", "right"]


def generate_report(start_time, end_time, total_subjects, output_path):
    duration = round((end_time - start_time), 2)
    ram_used = round((psutil.virtual_memory().total - psutil.virtual_memory().available) / (1024**3), 2)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Résumé d'Exécution - DPFStar (dhcpSym uniquement)", ln=True, align="C")

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
        f"Nombre de fichiers traités : {total_subjects}"
    ]

    for line in lines:
        pdf.cell(0, 10, line, ln=True)

    pdf.output(output_path)


def process_surface_file(file_path, log_file, error_log_file):
    try:
        # Extraction week-X, hemi-X depuis le chemin
        parts = file_path.split(os.sep)
        week_id = next((p for p in parts if p.startswith("week-")), "unknown_week")
        hemi_folder = parts[-3]  # hemi_left or hemi_right

        # Définir chemin de sortie
        out_dir = output_root
        os.makedirs(out_dir, exist_ok=True)

        # Lancer le calcul
        compute_dpfstar.main(file_path, output_dir=out_dir, display=False)

        with open(log_file, 'a') as log:
            log.write(f"{week_id}/{hemi_folder} : traitement OK - {os.path.basename(file_path)}\n")
        print(f"{week_id}/{hemi_folder} : traitement terminé")

    except Exception as e:
        with open(error_log_file, 'a') as errlog:
            errlog.write(f"Erreur avec {file_path}: {str(e)}\n")
            errlog.write(traceback.format_exc() + "\n")
        print(f"Erreur : {file_path}")


def main():
    start_time = time.time()
    all_files = []

    # Chercher tous les fichiers wm.gii en dhcpSym uniquement
    for week_dir in os.listdir(path_dataset):
        if not week_dir.startswith("week-") or week_dir == "week-to-40-registration":
            continue

        for hemi in hemi_id:
            search_path = os.path.join(path_dataset, week_dir, f"hemi_{hemi}", "dhcpSym", f"{week_dir}_hemi-{hemi}_space-dhcpSym_dens-32k_wm.surf.gii")
            matches = glob(search_path)
            if matches:
                all_files.extend(matches)

    total_files = len(all_files)

    # Préparer logs
    log_file = os.path.join(output_root, "log_execution.txt")
    error_log_file = os.path.join(output_root, "log_erreurs.txt")
    with open(log_file, 'w') as f:
        f.write("Log des traitements DPFStar - dhcpSym\n\n")
    with open(error_log_file, 'w') as f:
        f.write("Log des erreurs DPFStar - dhcpSym\n\n")

    # Traitement parallèle
    worker = partial(process_surface_file, log_file=log_file, error_log_file=error_log_file)
    with ProcessPoolExecutor() as executor:
        executor.map(worker, all_files)

    end_time = time.time()
    report_path = os.path.join(output_root, "resume_execution_dpfstar.pdf")
    generate_report(start_time, end_time, total_files, report_path)
    print(f"\n Rapport d'exécution généré : {report_path}")


if __name__ == "__main__":
    main()
