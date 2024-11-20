import os
import argparse

from functions import rw as sio
from functions import texture as stex
from functions import curvature as scurv 
import config as cfg  # Import du chemin depuis config.py

def save_curvature(K1, K2, Kmean, name_subject):
    # Création du dossier Parcelizer/named_subject
    save_folder = os.path.join(cfg.WD_FOLDER, name_subject)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Sauvegarde dans le dossier avec les noms K1.gii, K2.gii, et Kmean.gii
    sio.write_texture(stex.TextureND(darray=K1), os.path.join(save_folder, cfg.K1))
    sio.write_texture(stex.TextureND(darray=K2), os.path.join(save_folder, cfg.K2))
    sio.write_texture(stex.TextureND(darray=Kmean), os.path.join(save_folder, cfg.KMEAN))

def compute_curvature(mesh):
    # Calcul des composantes principales
    PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = scurv.curvatures_and_derivatives(mesh)
    K1 = PrincipalCurvatures[0, :]
    K2 = PrincipalCurvatures[1, :]
    # Calcul des courbures moyennes
    Kmean = 0.5 * (K1 + K2)
    return K1, K2, Kmean

def main(mesh_path):
    # Lecture du maillage
    mesh = sio.load_mesh(mesh_path)
    
    # Calcul de la courbure
    K1, K2, Kmean = compute_curvature(mesh)
    
    # Extraction du nom du fichier sans l'extension .gii
    name_subject = os.path.basename(mesh_path).replace('.gii', '')
    
    # Sauvegarde des résultats
    save_curvature(K1, K2, Kmean, name_subject)

if __name__ == "__main__":
    # Parse des arguments de la ligne de commande
    parser = argparse.ArgumentParser(description="Calculer et sauvegarder la courbure d'un maillage.")
    parser.add_argument("mesh_path", type=str, help="Chemin vers le fichier maillage .gii")
    args = parser.parse_args()

    # Appel de la fonction principale avec le chemin du maillage
    main(args.mesh_path)