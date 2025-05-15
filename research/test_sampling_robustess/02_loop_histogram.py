from research.tools import rw
import numpy as np
import sys
import os
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from research.tools.data_manager import list_files as dm 
import research.test_sampling_robustess.path_manager as pm


def plot_density_curves(dstar_list, res_list, subject_name, bins=40, save_dir="outputs/figures"):
    plt.figure(figsize=(10, 6))

    for dstar, res in zip(dstar_list, res_list):
        dstar = dstar[~np.isnan(dstar)]  # Nettoyage NaN
        kde = gaussian_kde(dstar)
        x_vals = np.linspace(np.min(dstar), np.max(dstar), 1000)
        y_vals = kde(x_vals)
        plt.plot(x_vals, y_vals, label=f"Résolution {res}", linewidth=2)

    plt.title(f"Courbes de densité DPF* : {subject_name}")
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


# Path manager and variables
wd = pm.WD_FOLDER
res_list = pm.SAMPLING_LIST
data_folder = os.path.join(wd, "data_test_sampling_robustess")
folder_dstar = os.path.join(data_folder, "textures")
gii_extension = ".gii"
bins = 40

# Loop through each subject in the list
subdataset_path = os.path.join(data_folder, "dataset")
files = dm.list_files(subdataset_path)
print("\n".join(files))


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

    plot_density_curves(dstar_list, res_list, subject_name=mesh_name, bins=bins)

    # Statistical tests
    #hist_100, _ = np.histogram(dstar_100, bins=bins, density=False)
    #hist_75, _ = np.histogram(dstar_75, bins=bins, density=False)
    #hist_50, _ = np.histogram(dstar_50, bins=bins, density=False)
    #hist_25, _ = np.histogram(dstar_25, bins=bins, density=False)

    #histograms = [hist_100, hist_75, hist_50, hist_25]

    #result_table = st.compare_histograms(histograms)

    # Display results
    #atools.display_dataframe_to_user(name="Résultats Comparaison Histogrammes", dataframe=result_table)

    # Interpretation and conclusion
    #st.interpret_results(result_table)