import numpy as np
import os
from research.tools import rw
from app.functions import texture as stex

def generate_cumulative_depth_maps(sub, ses, 
                                   dHCP_folder, 
                                   depth_map_folder, 
                                   output_folder,
                                   hemisphere='left',
                                   n_levels=50):
    """
    Génère des cartes cumulatives de profondeur pour les sulci et gyri.

    Parameters:
        sub (str): ID du sujet (ex: 'CC00576XX16')
        ses (str): Session (ex: '163200')
        dHCP_folder (str): Chemin vers les données dHCP
        depth_map_folder (str): Dossier contenant la carte de profondeur
        output_folder (str): Dossier de sauvegarde
        hemisphere (str): 'left' ou 'right'
        n_levels (int): Nombre de niveaux de discrétisation
    """

    # Chemins
    mesh_folder = os.path.join(dHCP_folder, f"sub-{sub}", f"ses-{ses}", "anat")
    label_file = f"sub-{sub}_ses-{ses}_hemi-{hemisphere}_desc-drawem_dseg.label.gii"
    depth_file = f"{sub}_{ses}_dpfstar500.gii"

    label_path = os.path.join(mesh_folder, label_file)
    depth_path = os.path.join(depth_map_folder, depth_file)

    # Chargement des données
    cortex_mask = rw.read_gii_file(label_path).astype(bool)
    depth = rw.read_gii_file(depth_path)

    # Discrétisation
    min_depth = np.min(depth[cortex_mask])
    max_depth = np.max(depth[cortex_mask])
    depth[~cortex_mask] = max_depth
    bins = np.linspace(min_depth, max_depth, n_levels)
    isolines = np.digitize(depth, bins=bins)

    # Sulci : zones profondes (cumulatif vers l'intérieur)
    isolines[~cortex_mask] = n_levels + 10
    cumulative_sulci = []
    for level in range(1, n_levels + 1):
        mask = (isolines <= level).astype(float)
        cumulative_sulci.append(mask)

    # Gyri : zones saillantes (cumulatif vers l'extérieur)
    isolines[~cortex_mask] = -10
    cumulative_gyri = []
    for level in range(1, n_levels + 1):
        mask = np.ones(len(depth))
        mask[isolines <= level] = 0
        cumulative_gyri.append(mask)

    # Sauvegarde
    os.makedirs(output_folder, exist_ok=True)
    sulci_path = os.path.join(output_folder, f"{sub}_{ses}_cumulative_sulci.gii")
    gyri_path = os.path.join(output_folder, f"{sub}_{ses}_cumulative_gyri.gii")

    rw.write_texture(stex.TextureND(darray=cumulative_sulci), sulci_path)
    rw.write_texture(stex.TextureND(darray=cumulative_gyri), gyri_path)

    print(f"Sauvegardé :\n - {sulci_path}\n - {gyri_path}")


if __name__ == "__main__":
    # Paramètres
    wd = "/home/maxime/callisto/repo/paper_sulcal_depth"
    dHCP_folder = "/media/maxime/Expansion/rel3_dHCP"
    depth_map_folder = os.path.join(wd, "data_EXP3/result_EXP3/dpfstar500")
    output_folder = os.path.join(wd, "data_EXP4/result_EXP4", "CC00576XX16_163200")

    generate_cumulative_depth_maps(
        sub="CC00576XX16",
        ses="163200",
        dHCP_folder=dHCP_folder,
        depth_map_folder=depth_map_folder,
        output_folder=output_folder,
        hemisphere="left",
        n_levels=50
    )
