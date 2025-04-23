import os
import argparse
from app.functions import rw 
from app.functions import texture as stex
from app.functions import dpfstar
import app.config as cfg  # Import du chemin depuis config.py

def save_dpfstar(dpfstar, name_subject, output_dir=None):
    # Utiliser le dossier de sortie spécifié ou cfg par défaut
    save_folder_base = output_dir if output_dir is not None else cfg.WD_FOLDER
    save_folder = os.path.join(save_folder_base, name_subject)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    print("DPF enregistrée dans :", save_folder)
    rw.write_texture(stex.TextureND(darray=dpfstar), os.path.join(save_folder, cfg.DPFSTAR))


def call_compute_curvature(mesh_path):
    os.system(f"python -m app.compute_curvature {mesh_path}")

def call_visualizer(mesh_path, texture_path):
    os.system(f"python -m app.visualizer {mesh_path} --texture {texture_path}")

def main(mesh_path, curvature=None, display=False, output_dir=None):
    # Lecture du maillage
    mesh = rw.load_mesh(mesh_path)
    name_subject = os.path.basename(mesh_path).replace('.gii', '')

    # Dossier utilisé pour chercher/sauver des fichiers associés
    save_folder_base = output_dir if output_dir is not None else cfg.WD_FOLDER
    save_subject_path = os.path.join(save_folder_base, name_subject)

    # Lecture ou calcul de la courbure
    if curvature is None:
        kmean_path = os.path.join(save_subject_path, cfg.KMEAN)
        if os.path.exists(kmean_path):
            Kmean = rw.read_gii_file(kmean_path)
        else:
            print("Calcul de la courbure car aucune courbure fournie.")
            call_compute_curvature(mesh_path)
            Kmean = rw.read_gii_file(kmean_path)
    else:
        Kmean = rw.read_gii_file(curvature)

    # Calcul et sauvegarde DPF
    dp = dpfstar.dpf_star(mesh, Kmean)
    save_dpfstar(dp, name_subject, output_dir=output_dir)

    # Affichage optionnel
    if display:
        try:
            texture_path = os.path.join(save_subject_path, cfg.DPFSTAR)
            call_visualizer(mesh_path, texture_path)
        except Exception as e:
            print("Erreur lors de la visualisation :", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculer et sauvegarder la dpf-star d'un maillage.")
    parser.add_argument("mesh_path", type=str, help="Chemin vers le fichier maillage .gii")
    parser.add_argument("--curvature", type=str, help="Chemin vers la courbure du maillage", default=None)
    parser.add_argument("--display", action="store_true", help="Utiliser --display pour afficher le résultat")
    parser.add_argument("--outputdir", type=str, help="Dossier de sortie pour sauvegarder les résultats", default=None)

    args = parser.parse_args()
    main(args.mesh_path, curvature=args.curvature, display=args.display, output_dir=args.outputdir)
