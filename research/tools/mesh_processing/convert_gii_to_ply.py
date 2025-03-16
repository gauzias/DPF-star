import nibabel as nib
import numpy as np
import trimesh
import sys

def convert_gii_to_ply(input_gii_path, output_ply_path):
    """
    Convertit un fichier GIfTI (.gii) en PLY (.ply).

    :param input_gii_path: Chemin du fichier GIfTI d'entrée
    :param output_ply_path: Chemin du fichier PLY de sortie
    """
    # Charger le fichier GIfTI
    gii = nib.load(input_gii_path)
    
    # Vérifier que le fichier contient bien des données de maillage
    if len(gii.darrays) < 2:
        raise ValueError("Le fichier GIfTI ne contient pas suffisamment de données (sommets et faces).")

    # Extraction des vertices et des faces
    verts = np.array(gii.darrays[0].data, dtype=np.float32)  # Coordonnées des sommets (x, y, z)
    faces = np.array(gii.darrays[1].data, dtype=np.int32)  # Connexions des faces (triangles)

    # Vérification des dimensions
    if verts.shape[1] != 3:
        raise ValueError("Les sommets du fichier GIfTI ne sont pas en format (N, 3).")

    if faces.shape[1] != 3:
        raise ValueError("Les faces du fichier GIfTI ne sont pas des triangles.")

    # Création du maillage avec Trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    # Sauvegarde en PLY
    mesh.export(output_ply_path, file_type="ply")
    print(f"Conversion terminée : {output_ply_path}")

# Exemple d'utilisation
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_gii_to_ply.py input.gii output.ply")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        convert_gii_to_ply(input_file, output_file)
