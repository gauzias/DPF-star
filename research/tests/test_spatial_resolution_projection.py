from research.tools import interpolation_meshes as inter
from research.tools import rw 
from app.functions import texture as stex
import nibabel as nib
import numpy as np
import trimesh
from scipy.spatial import KDTree
import app.config as cfg  # Import du chemin depuis config.py
import os

# Charger les maillages GIFTI
path_high_res_mesh = "D:/Callisto/repo/DPF-star/meshes/mesh.gii"
path_low_res_mesh = "D:/Callisto/repo/DPF-star/meshes/mesh_decimation_05.gii"

mesh_high = rw.load_mesh(path_high_res_mesh)
mesh_low = rw.load_mesh(path_low_res_mesh)


# charger la DPF-star
path_high_depth = "D:/Callisto/wd_dpfstar/mesh/dpfstar.gii"
path_low_depth = "D:/Callisto/wd_dpfstar/mesh_decimation_05/dpfstar.gii"
high_depth = rw.read_gii_file(path_high_depth)
low_depth = rw.read_gii_file(path_low_depth)

# Mapper la profondeur du maillage haute résolution sur le maillage basse résolution
mapped_depth_low = inter.map_low_to_high(mesh_high, mesh_low, high_depth)


# sauvegarder mapped_depth_low
save_folder = os.path.join(cfg.WD_FOLDER, "mesh_decimation_05")
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
rw.write_texture(stex.TextureND(darray=mapped_depth_low), os.path.join(save_folder, "mapped_dpfstar.gii"))


# calcul des erreurs globals: 

valid_indices = ~np.isnan(mapped_depth_low)
mse = np.mean((low_depth[valid_indices] - mapped_depth_low[valid_indices])**2)
nrmse = np.sqrt(mse) / (np.max(low_depth) - np.min(low_depth))
mae = np.mean(np.abs(low_depth[valid_indices] - mapped_depth_low[valid_indices]))

print(f"MSE: {mse}, NRMSE: {nrmse}, MAE: {mae}")
# Afficher la comparaison
import matplotlib.pyplot as plt

plt.scatter(mapped_depth_low, high_depth[:len(mapped_depth_low)], alpha=0.5)
plt.xlabel("Profondeur interpolée sur basse résolution")
plt.ylabel("Profondeur haute résolution")
plt.title("Comparaison des profondeurs interpolées")
plt.show()
