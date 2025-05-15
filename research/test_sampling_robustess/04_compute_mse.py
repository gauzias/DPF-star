from research.tools import rw
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import app.config as cfg
import ace_tools_open as atools
from research.tools.data_manager import list_files as dm

# Configuration
folder_mapped_dstar = "E:/research_dpfstar/results_rel3_dhcp"
folder_dstar = "E:/research_dpfstar/wd_dpfstar"
folder_mesh = "E:/research_dpfstar/data_repo_dpfstar/data_test_resolution"
gii_extension = ".gii"
res_source = 100
res_target_list = [75, 50, 25]
results = []

print("Recherche des fichiers à résolution 100...")
files = dm.list_files(os.path.join(folder_mesh, "subdataset"))
print(f"{len(files)} fichiers trouvés\n")

for mesh_path in files:
    mesh_name = dm.get_filename_without_extension(mesh_path, gii_extension)
    base_name = re.sub(f"_res{res_source}$", "", mesh_name)

    print(f"\n Évaluation pour : {base_name}")

    for res_target in res_target_list:
        path_low_dpf = os.path.join(folder_dstar, f"{base_name}_res{res_target}", "dpfstar.gii")
        path_mapped = os.path.join(folder_mapped_dstar, f"{base_name}_res{res_target}", "mapped_dpfstar.gii")

        print(f"Chargement :\n - Référence : {path_low_dpf}\n - Interpolé : {path_mapped}")
        low_depth = rw.read_gii_file(path_low_dpf)
        mapped_depth = rw.read_gii_file(path_mapped)

        valid = ~np.isnan(mapped_depth)
        mse = np.mean((low_depth[valid] - mapped_depth[valid])**2)
        nrmse = np.sqrt(mse) / np.sqrt(np.mean(mapped_depth[valid] ** 2))
        mae = np.mean(np.abs(low_depth[valid] - mapped_depth[valid]))

        results.append({
            "Sujet": base_name,
            "Résolution cible": res_target,
            "MSE": mse,
            "NRMSE": nrmse,
            "MAE": mae
        })

        print(f"MSE={mse:.5f}, NRMSE={nrmse:.5f}, MAE={mae:.5f}")

# Résultats finaux
df_results = pd.DataFrame(results)
atools.display_dataframe_to_user(name="Évaluation DPF* à partir des fichiers .gii", dataframe=df_results)

# Sauvegarde CSV
csv_path = os.path.join(cfg.WD_FOLDER, "interpolation_errors_summary_reloaded.csv")
df_results.to_csv(csv_path, index=False)
print(f"\n Résumé des erreurs sauvegardé dans : {csv_path}")

# Visualisation en boxplots
sns.set(style="whitegrid")
save_dir = os.path.join(cfg.WD_FOLDER, "figures_eval")
os.makedirs(save_dir, exist_ok=True)

for metric in ["MSE", "NRMSE", "MAE"]:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_results, x="Résolution cible", y=metric, palette="Set2")
    plt.title(f"Distribution de {metric} par résolution cible", fontsize=14)
    plt.xlabel("Résolution cible", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.tight_layout()

    # Enregistrement de la figure
    fig_path = os.path.join(save_dir, f"boxplot_{metric}_reloaded.png")
    plt.savefig(fig_path, dpi=300)
    print(f"Figure sauvegardée : {fig_path}")

    plt.show()
