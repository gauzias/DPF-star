
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

def check_kki2009(filename):
    match = re.search(r'sub-([A-Za-z0-9]+)', filename)
    if match and match.group(1) == "KKI2009":
        return True
    return False




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
data_folder  = "D:/Callisto/data/data_repo_dpfstar/data_test_resolution"
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
    if dsetKKI : 
          # load mask cortex
        path_interH = os.path.join(data_folder, "interH_KKI2009", )
        cortex = sio.load_texture(os.path.join('/media/maxime/Expansion/FS_database_KKI_test_retest_FS6.0',
                                 sub + '_' + ses, 'label', 'mask_interH.gii')).darray[0]
        cortex = np.invert(cortex.astype(bool))
    else:

        cortex = sio.load_texture(os.path.join('/media/maxime/Expansion/rel3_dHCP/', 'sub-' + sub, 'ses-' + ses, 'anat',
                                               'sub-' + sub + '_ses-' + ses + '_hemi-left_desc-drawem_dseg.label.gii'))
        cortex = cortex.darray[0].astype(bool)

    
  
    # path dpf star

    dstar_list = []
    res_list = [100,75,50,25,10]
    folder_dstar = "D:/Callisto/wd_dpfstar/"
    for res in res_list : 
        name_mesh_decimate = mesh_name + '_res' + str(res)
        path_dstar = os.path.join(folder_dstar, name_mesh_decimate, "dpfstar.gii") 
        dstar = rw.read_gii_file(path_dstar)
        dstar_list.append(dstar)
    
    dstar_100 = dstar_list[0]
    dstar_75 = dstar_list[1]
    dstar_50 = dstar_list[2]
    dstar_25 = dstar_list[3]
    dstar_10 = dstar_list[4]

        
    plot_histograms(dstar_100, dstar_75, dstar_50, dstar_25, dstar_10, bins=bins)
    
    # Test statistiques
    
    hist_100, _ = np.histogram(dstar_100, bins=bins, density=False)
    hist_75, _ = np.histogram(dstar_75, bins=bins, density=False)
    hist_50, _ = np.histogram(dstar_50, bins=bins, density=False)
    hist_25, _ = np.histogram(dstar_25, bins=bins, density=False)
    hist_10, _ = np.histogram(dstar_10, bins=bins, density=False)
    
    histograms =  [hist_100, hist_75, hist_50, hist_25, hist_10]

    result_table = st.compare_histograms(histograms)

    # Affichage des résultats
    atools.display_dataframe_to_user(name="Résultats Comparaison Histogrammes", dataframe=result_table)
    
    # Interprétation et conclusion
    st.interpret_results(result_table)






