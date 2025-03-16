import trimesh
import nibabel as nib
import numpy as np

def convert_ply_to_gii(input_ply, output_gii):
    # Charger le maillage PLY avec trimesh
    mesh = trimesh.load_mesh(input_ply)

    # Récupérer les sommets et les faces
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)

    # Créer un objet GIfTI pour les sommets et les faces
    gii_data = nib.GiftiImage()
    gii_data.add_gifti_data_array(nib.gifti.GiftiDataArray(vertices, intent=nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']))
    gii_data.add_gifti_data_array(nib.gifti.GiftiDataArray(faces, intent=nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']))

    # Sauvegarder en fichier GIfTI
    nib.save(gii_data, output_gii)
    print(f"Conversion terminée : {output_gii}")

# Exemple d'utilisation
input_ply = "D:/Callisto/repo/meshprocessing/samples/mesh_decimation_05.ply"
output_gii = "D:/Callisto/repo/meshprocessing/samples/mesh_decimation_05.gii"

convert_ply_to_gii(input_ply, output_gii)
