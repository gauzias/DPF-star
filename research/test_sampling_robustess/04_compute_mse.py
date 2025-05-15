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
import research.test_sampling_robustess.path_manager as pm

# Configuration
wd = pm.WD_FOLDER
data_folder = os.path.join(wd, "data_test_sampling_robustess")
folder_dstar = os.path.join(data_folder, "textures")
folder_mapped_dstar = os.path.join(data_folder, "textures")
folder_mesh = os.path.join(data_folder, "sampled_meshes")
gii_extension = ".gii"
res_source = 100
res_target_list = [75, 50, 25]

results = []

print("Searching for resolution 100 files...")
subdataset_path = os.path.join(folder_mesh, "subdataset")
files = dm.list_files(subdataset_path)
print(f"{len(files)} files found\n")

for mesh_path in files:
    mesh_name = dm.get_filename_without_extension(mesh_path, gii_extension)
    base_name = re.sub(f"_res{res_source}$", "", mesh_name)

    print(f"\nEvaluation for: {base_name}")

    for res_target in res_target_list:
        path_low_dpf = os.path.join(folder_dstar, f"{base_name}_res{res_target}", "dpfstar.gii")
        path_mapped = os.path.join(folder_mapped_dstar, f"{base_name}_res{res_target}", "mapped_dpfstar.gii")

        print(f"Loading:\n - Reference: {path_low_dpf}\n - Interpolated: {path_mapped}")
        low_depth = rw.read_gii_file(path_low_dpf)
        mapped_depth = rw.read_gii_file(path_mapped)

        valid = ~np.isnan(mapped_depth)
        mse = np.mean((low_depth[valid] - mapped_depth[valid])**2)
        nrmse = np.sqrt(mse) / np.std(mapped_depth[valid])
        mae = np.mean(np.abs(low_depth[valid] - mapped_depth[valid]))

        results.append({
            "Sujet": base_name,
            "Résolution cible": res_target,
            "MSE": mse,
            "NRMSE": nrmse,
            "MAE": mae
        })

        print(f"MSE={mse:.5f}, NRMSE={nrmse:.5f}, MAE={mae:.5f}")

# Final results
df_results = pd.DataFrame(results)
atools.display_dataframe_to_user(name="Évaluation DPF* à partir des fichiers .gii", dataframe=df_results)

# Save CSV
stats_folder = os.path.join(data_folder, "stats")
os.makedirs(stats_folder, exist_ok=True)
csv_path = os.path.join(stats_folder, "interpolation_errors_summary_reloaded.csv")
df_results.to_csv(csv_path, index=False)
print(f"\nError summary saved to: {csv_path}")

# Boxplot visualization
sns.set(style="whitegrid")

for metric in ["MSE", "NRMSE", "MAE"]:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_results, x="Résolution cible", y=metric, palette="Set2")
    plt.title(f"Distribution of {metric} by target resolution", fontsize=14)
    plt.xlabel("Target resolution", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(stats_folder, f"boxplot_{metric}.png")
    plt.savefig(fig_path, dpi=300)
    print(f"Figure saved: {fig_path}")

    plt.show()
