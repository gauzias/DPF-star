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
import research.test_sampling_robustess.path_manager as pm

# Configuration
wd = pm.WD_FOLDER
data_folder = os.path.join(wd, "data_test_sampling_robustess")
folder_dstar = os.path.join(data_folder, "textures")
folder_mesh = os.path.join(data_folder, "sampled_meshes")
gii_extension = ".gii"
res_source = 100
res_target_list = [75, 50, 25]


print("Searching for resolution 100 files...")
dataset_path = os.path.join(data_folder, "dataset")
files = dm.list_files(dataset_path)
print(f"{len(files)} files found\n")

for mesh_path in files:
    mesh_name = dm.get_filename_without_extension(mesh_path, gii_extension)
    base_name = re.sub(f"_res{res_source}$", "", mesh_name)

    print(f"\nProcessing subject: {base_name}")

    path_high_mesh = os.path.join(folder_mesh, base_name, f"{base_name}_res{res_source}.gii")
    path_high_dpf = os.path.join(folder_dstar, f"{base_name}_res{res_source}", "dpfstar.gii")
    print(f"High resolution files:\n - Mesh: {path_high_mesh}\n - DPF*: {path_high_dpf}")

    print(f"Loading mesh and DPF* at resolution {res_source}")
    mesh_high = rw.load_mesh(path_high_mesh)
    high_depth = rw.read_gii_file(path_high_dpf)

    for res_target in res_target_list:
        print(f"\nInterpolating to resolution {res_target}")
        path_low_mesh = os.path.join(folder_mesh, base_name, f"{base_name}_res{res_target}.gii")
        path_low_dpf = os.path.join(folder_dstar, f"{base_name}_res{res_target}", "dpfstar.gii")

        print(f"Low resolution files:\n - Mesh: {path_low_mesh}\n - DPF*: {path_low_dpf}")
        mesh_low = rw.load_mesh(path_low_mesh)
        low_depth = rw.read_gii_file(path_low_dpf)

        print("Interpolating...")
        mapped_depth = inter.map_low_to_high(mesh_high, mesh_low, high_depth)

        save_folder = os.path.join(folder_dstar, f"{base_name}_res{res_target}")
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, "mapped_dpfstar.gii")
        rw.write_texture(stex.TextureND(darray=mapped_depth), save_path)
        print(f"Interpolated texture saved at: {save_path}")
