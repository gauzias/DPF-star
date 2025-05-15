from research.tools import interpolation_meshes as inter
from research.tools import rw 
from app.functions import texture as stex
import numpy as np
import os
import re
import pandas as pd
from research.tools.data_manager import list_files as dm 
import app.config as cfg
import ace_tools_open as atools

# Configuration
data_folder = "E:/research_dpfstar/data_repo_dpfstar/data_test_resolution"
folder_dstar = "E:/research_dpfstar/wd_dpfstar/"
folder_mesh = "E:/research_dpfstar/data_repo_dpfstar/data_test_resolution"
gii_extension = ".gii"
res_source = 100
res_target_list = [75, 50, 25]

results = []

print("Recherche des fichiers à résolution 100...")
files = dm.list_files(os.path.join(data_folder, "subdataset"))
print(f"{len(files)} fichiers trouvés\n")

for mesh_path in files:

    mesh_name = dm.get_filename_without_extension(mesh_path, gii_extension)
    base_name = re.sub(f"_res{res_source}$", "", mesh_name)

    print(f"\n Traitement du sujet : {base_name}")

    path_high_mesh = os.path.join(folder_mesh, f"{base_name}", f"{base_name}_res{res_source}.gii")
    path_high_dpf = os.path.join(folder_dstar, f"{base_name}_res{res_source}", "dpfstar.gii")
    print(f"Fichiers haute résolution :\n - Mesh : {path_high_mesh}\n - DPF* : {path_high_dpf}")

    print(f"Chargement du maillage et DPF* à résolution {res_source}")
    mesh_high = rw.load_mesh(path_high_mesh)
    high_depth = rw.read_gii_file(path_high_dpf)

    for res_target in res_target_list:
        print(f"\n Interpolation vers résolution {res_target}")
        path_low_mesh = os.path.join(folder_mesh, f"{base_name}", f"{base_name}_res{res_target}.gii")
        path_low_dpf = os.path.join(folder_dstar, f"{base_name}_res{res_target}", "dpfstar.gii")

        print(f"Fichiers basse résolution :\n - Mesh : {path_low_mesh}\n - DPF* : {path_low_dpf}")
        mesh_low = rw.load_mesh(path_low_mesh)
        low_depth = rw.read_gii_file(path_low_dpf)

        print("Interpolation en cours...")
        mapped_depth = inter.map_low_to_high(mesh_high, mesh_low, high_depth)

        save_folder = os.path.join(cfg.WD_FOLDER, f"{base_name}_res{res_target}")
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, "mapped_dpfstar.gii")
        rw.write_texture(stex.TextureND(darray=mapped_depth), save_path)
        print(f"Texture interpolée sauvegardée : {save_path}")

        valid = ~np.isnan(mapped_depth)
        mse = np.mean((low_depth[valid] - mapped_depth[valid])**2)
        nrmse = np.sqrt(mse) / (np.max(low_depth) - np.min(low_depth))
        mae = np.mean(np.abs(low_depth[valid] - mapped_depth[valid]))

        results.append({
            "Sujet": base_name,
            "Résolution cible": res_target,
            "MSE": mse,
            "NRMSE": nrmse,
            "MAE": mae
        })

        print(f" Erreurs : MSE={mse:.5f}, NRMSE={nrmse:.5f}, MAE={mae:.5f}")

# Résultats finaux
df_results = pd.DataFrame(results)
atools.display_dataframe_to_user(name="Erreurs interpolation DPF*", dataframe=df_results)

csv_path = os.path.join(cfg.WD_FOLDER, "interpolation_errors_summary.csv")
df_results.to_csv(csv_path, index=False)
print(f"\n Résumé des erreurs sauvegardé dans : {csv_path}")
