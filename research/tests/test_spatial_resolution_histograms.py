
from research.tools import rw
from research.tools import dpfstar
from research.tools import dynamic_histogramm as viz
from research.tools import compare_histograms as st
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from app.compute_dpfstar import main as dpfstar

# Pour lancer ce code : 
# 1. se mettre dans l'environement conda DPF-star
# 2. lancer a partir de D:\Callisto\repo\DPF-star : python -m research.tests.test_spatial_resolution_robustness


# mesh path 

path_mesh_1 = "D:/Callisto/repo/DPF-star/meshes/mesh.gii"
path_mesh_05 = "D:/Callisto/repo/DPF-star/meshes/mesh_decimation_05.gii"
path_mesh_02 = "D:/Callisto/repo/DPF-star/meshes/mesh_decimation_02.gii"

# test load samples
mesh_1 = rw.load_mesh("D:/Callisto/repo/DPF-star/meshes/mesh.gii") # original mesh
mesh_05 = rw.load_mesh("D:/Callisto/repo/DPF-star/meshes/mesh_decimation_05.gii") # decimated mesh 0.5 of original
mesh_02 = rw.load_mesh("D:/Callisto/repo/DPF-star/meshes/mesh_decimation_02.gii") # decimated mesh 0.2 of original

print("Nombre sommet mesh 1 :", len(mesh_1.vertices))
print("Nombre sommet mesh 0.5 :",len(mesh_05.vertices))
print("Nombre sommet mesh 0.2 :",len(mesh_02.vertices))

# Compute DPF-star for each samples
dpfstar(path_mesh_1, curvature=None, display=False)
dpfstar(path_mesh_05, curvature=None, display=False)
dpfstar(path_mesh_02, curvature=None, display=False)

# pour visualiser :  
# python -m app.visualizer D:/Callisto/repo/DPF-star/meshes/mesh_decimation_05.gii --texture D:/Callisto/wd_dpfstar/mesh_decimation_05/dpfstar.gii

# Display the histogram for each computed dpf-star

path_dstar_1 = "D:/Callisto/wd_dpfstar/mesh/dpfstar.gii"
path_dstar_05 = "D:/Callisto/wd_dpfstar/mesh_decimation_05/dpfstar.gii"
path_dstar_02 = "D:/Callisto/wd_dpfstar/mesh_decimation_02/dpfstar.gii"

dstar_1 = rw.read_gii_file(path_dstar_1)
dstar_05 = rw.read_gii_file(path_dstar_05)
dstar_02 = rw.read_gii_file(path_dstar_02)


viz.plot_histograms(dstar_1, dstar_05, dstar_02)


# Test statistiques
hist_1, _ = np.histogram(dstar_1, bins=23, density=False)
hist_05, _ = np.histogram(dstar_05, bins=23, density=False)
hist_02, _ = np.histogram(dstar_02, bins=23, density=False)
histograms =  [hist_1, hist_05, hist_02]

result_table = st.compare_histograms(histograms)

# Affichage des résultats
import ace_tools_open as atools
atools.display_dataframe_to_user(name="Résultats Comparaison Histogrammes", dataframe=result_table)

# Interprétation et conclusion
st.interpret_results(result_table)