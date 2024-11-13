import os
import argparse
from functions import rw 
from functions import texture as stex
from functions import dpfstar
import config as cfg  # Import du chemin depuis config.py

def save_dpfstar(dpfstar, name_subject):
    # Création du dossier wd/named_subject
    save_folder = os.path.join(cfg.WD_FOLDER, name_subject)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Sauvegarde dans le dossier avec le nom dpfstar.gii 
    rw.write_texture(stex.TextureND(darray=dpfstar), os.path.join(save_folder, cfg.DPFSTAR))


def call_compute_curvature(mesh_path):
    os.system(f"python .\compute_curvature.py {mesh_path}")

def main(mesh_path, curvature=None):
    # Lecture du maillage
    mesh = rw.load_mesh(mesh_path)

    # Extraction du nom du fichier sans l'extension .gii
    name_subject = os.path.basename(mesh_path).replace('.gii', '')

    if curvature==None:
        try:
            Kmean = rw.read_gii_file(os.path.join(cfg.WD_FOLDER, name_subject, cfg.KMEAN))
        except:
            print("computation of curvature")
            call_compute_curvature(mesh_path)
            Kmean = rw.read_gii_file(os.path.join(cfg.WD_FOLDER, name_subject, cfg.KMEAN))
    
    # Calcul de la dpfstar
    dstar = dpfstar.dpf_star(mesh, Kmean)

    # Sauvegarde de la dpfstar
    save_dpfstar(dstar, name_subject)



if __name__ == "__main__":
    # Parse des arguments de la ligne de commande
    parser = argparse.ArgumentParser(description="Calculer et sauvegarder la cdpfstar d'un maillage.")
    parser.add_argument("mesh_path", type=str, help="Chemin vers le fichier maillage .gii")
    parser.add_argument('--curvature', type=str, help="Chemin vers le courbure du maillage", default=None)
    args = parser.parse_args()

    # Appel de la fonction principale avec le chemin du maillage
    main(args.mesh_path)