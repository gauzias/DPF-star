from research.tools import rw
from research.tools.histogramm_analysis import compare_histograms as st
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from research.tools.data_manager import list_files as dm 
import ace_tools_open as atools
import research.test_sampling_robustess.path_manager as pm
from research.test_sampling_robustess.functions import tools_for_histogram as tfh

# Display the histogram for each computed dpf-star

# Path manager and variables
wd = pm.WD_FOLDER
res_list = pm.SAMPLING_LIST
data_folder = os.path.join(wd, "data_test_sampling_robusess")
folder_dstar = os.path.join(data_folder, "textures")
gii_extension = ".gii"
bins = 40

# Loop through each subject in the list
try:
    subdataset_path = os.path.join(data_folder, "dataset")
    files = dm.list_files(subdataset_path)
    print("\n".join(files))
except ValueError as e:
    print(e)

# Loop to compute dpf-star
for mesh_path in files:
    print(mesh_path)
    # Get filename without extension
    mesh_name = dm.get_filename_without_extension(mesh_path, gii_extension)

    # Path to DPF-star results
    dstar_list = []

    for res in res_list:
        name_mesh_decimate = mesh_name + "_res" + str(res)
        path_dstar = os.path.join(folder_dstar, name_mesh_decimate, "dpfstar.gii")
        dstar = rw.read_gii_file(path_dstar)
        dstar_list.append(dstar)

    dstar_100 = dstar_list[0]
    dstar_75 = dstar_list[1]
    dstar_50 = dstar_list[2]
    dstar_25 = dstar_list[3]

    tfh.plot_density_curves(dstar_list, res_list, subject_name=mesh_name, bins=bins)

    # Statistical tests
    hist_100, _ = np.histogram(dstar_100, bins=bins, density=False)
    hist_75, _ = np.histogram(dstar_75, bins=bins, density=False)
    hist_50, _ = np.histogram(dstar_50, bins=bins, density=False)
    hist_25, _ = np.histogram(dstar_25, bins=bins, density=False)

    histograms = [hist_100, hist_75, hist_50, hist_25]

    result_table = st.compare_histograms(histograms)

    # Display results
    atools.display_dataframe_to_user(name="Résultats Comparaison Histogrammes", dataframe=result_table)

    # Interpretation and conclusion
    st.interpret_results(result_table)