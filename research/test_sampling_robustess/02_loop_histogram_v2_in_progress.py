
from research.tools import rw
from research.tools.histogramm_analysis import compare_histograms as st
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import matplotlib.pyplot as plt
import math
from research.tools.data_manager import list_files as dm 
import ace_tools_open as atools
import re
from research.tools import rw 
from scipy.stats import gaussian_kde
import re



def extraire_id_session(nom_fichier):
    # Expression régulière adaptée à la nouvelle structure
    match = re.match(r"sub-[^_]+_([0-9]+)_ses-([^_]+)", nom_fichier)
    if match:
        identifiant = match.group(1)
        session = match.group(2)
        return identifiant, session
    else:
        raise ValueError("Le nom de fichier ne respecte pas le format attendu.")
    
def extraire_sub_ses(nom_fichier):
    # Utilisation d'une expression régulière pour extraire les champs
    match = re.match(r"sub-([a-zA-Z0-9]+)_ses-([0-9]+)", nom_fichier)
    if match:
        sub = match.group(1)
        ses = match.group(2)
        return sub, ses
    else:
        raise ValueError("Le nom de fichier ne respecte pas le format attendu.")



def check_kki2009(filename):
    match = re.search(r'sub-([A-Za-z0-9]+)', filename)
    if match and match.group(1) == "KKI2009":
        return True
    return False


def plot_density_curves(dstar_list, res_list, subject_name, bins=40, save_dir="outputs/figures"):
    plt.figure(figsize=(10, 6))

    for dstar, res in zip(dstar_list, res_list):
        dstar = dstar[~np.isnan(dstar)]  # Nettoyage NaN
        kde = gaussian_kde(dstar)
        x_vals = np.linspace(np.min(dstar), np.max(dstar), 1000)
        y_vals = kde(x_vals)
        plt.plot(x_vals, y_vals, label=f"Résolution {res}", linewidth=2)

    plt.title(f"Courbes de densité DPF* – {subject_name}")
    plt.xlabel("Valeur DPF*")
    plt.ylabel("Densité")
    plt.legend(title="Résolutions")
    plt.grid(True)
    plt.tight_layout()

    # Création du dossier de sauvegarde s’il n’existe pas
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{subject_name}_density_plot.png")
    plt.savefig(save_path)
    plt.close()

def plot_histograms(*lists, bins):
    n = len(lists)
    cols = min(5, n)  # Maximum de 5 subplots par ligne
    rows = math.ceil(n / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if n > 1 else [axes]  # Assurer une itération correcte
    
    for i, data in enumerate(lists):
        axes[i].hist(data, bins=bins, alpha=0.7, color='blue', edgecolor='black')
        axes[i].set_title(f'Histogram {i+1}')
    
    # Cacher les subplots inutilisés
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

# Display the histogram for each computed dpf-star

# path manager and variables
resolutions_list = [10, 25, 75, 100]
data_folder  = "E:/research_dpfstar/data_repo_dpfstar/data_test_resolution"
gii_extension = ".gii"
bins = 40

# boucle pour chaque sujet dans la liste : 
try:
    files = dm.list_files(os.path.join(data_folder, "subdataset"))
    print("\n".join(files))
except ValueError as e:
    print(e)

### Loop dcompute dpfstar

for mesh_path in files :
    print(mesh_path)
    # get name file
    mesh_name = dm.get_filename_without_extension(mesh_path, gii_extension)
    subject_folder = os.path.join(data_folder, mesh_name)
    dsetKKI =  check_kki2009(mesh_name)
    
    #if dsetKKI : 
        #sub, ses = extraire_id_session(mesh_name)
        # load mask cortex
        #cortex = rw.read_gii_file(os.path.join('E:/FS_database_KKI_test_retest_FS6.0',
        #                         'KKI_' + sub + '_' + ses, 'label', 'mask_interH.gii'))
        #cortex = np.invert(cortex.astype(bool))
        #mesh = rw.read_gii_file(os.path.join('E:/FS_database_KKI_test_retest_FS6.0', 'KKI_' + sub + '_' + ses, 'surf', 'lh.white.gii'))
        #cortex = np.ones(len(mesh))
        #cortex = cortex.astype(bool)
    #    print('next')
    #else:
        
    #    sub, ses = extraire_sub_ses(mesh_name)
    #    cortex = rw.read_gii_file(os.path.join('E:','rel3_dHCP', 'sub-' + sub, 'ses-' + ses, 'anat',
    #                                          'sub-' + sub + '_ses-' + ses + '_hemi-left_desc-drawem_dseg.label.gii'))
    #    cortex = cortex.astype(bool)

   

    # path dpf star

    dstar_list = []
    res_list = [100,75,50,25]
    folder_dstar = "E:/research_dpfstar/wd_dpfstar/"
    for res in res_list : 
        name_mesh_decimate = mesh_name + '_res' + str(res)
        path_dstar = os.path.join(folder_dstar, name_mesh_decimate, "dpfstar.gii") 
        dstar = rw.read_gii_file(path_dstar)
        #dstar = dstar[cortex]
        dstar_list.append(dstar)
    
    dstar_100 = dstar_list[0]
    dstar_75 = dstar_list[1]
    dstar_50 = dstar_list[2]
    dstar_25 = dstar_list[3]
    #dstar_10 = dstar_list[4]
    #plot_histograms(dstar_100, dstar_75, dstar_50, dstar_25, dstar_10, bins=bins)
    plot_density_curves(dstar_list, res_list, subject_name=mesh_name, bins=bins)
    
    # Test statistiques
    
    hist_100, _ = np.histogram(dstar_100, bins=bins, density=False)
    hist_75, _ = np.histogram(dstar_75, bins=bins, density=False)
    hist_50, _ = np.histogram(dstar_50, bins=bins, density=False)
    hist_25, _ = np.histogram(dstar_25, bins=bins, density=False)
    #hist_10, _ = np.histogram(dstar_10, bins=bins, density=False)
    
    #histograms =  [hist_100, hist_75, hist_50, hist_25, hist_10]
    histograms =  [hist_100, hist_75, hist_50, hist_25]

    result_table = st.compare_histograms(histograms)

    # Affichage des résultats
    atools.display_dataframe_to_user(name="Résultats Comparaison Histogrammes", dataframe=result_table)
    
    # Interprétation et conclusion
    st.interpret_results(result_table)






