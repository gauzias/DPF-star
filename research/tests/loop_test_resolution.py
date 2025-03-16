
import os
from research.tools.data_manager import list_files as dm 
from research.tools.mesh_processing import convert_gii_to_ply as mp
from research.tools.mesh_processing import filter_decimation as fd

# path manager

data_folder  = "D:/Callisto/data/data_repo_dpfstar/data_test_resolution"
input_extension = ".gii"
output_extension = ".ply"

# boucle pour chaque sujet dans la liste : 

directory_path = "D:/Callisto/data/data_repo_dpfstar/data_test_resolution"
try:
    files = dm.list_files(directory_path)
    print("\n".join(files))
except ValueError as e:
    print(e)

### faire trois resolution differentes

for mesh_path in files :
    # get name file
    mesh_name = dm.get_filename_without_extension(mesh_path, input_extension)
    mesh_res100_name = mesh_name + '_res100'
    mesh_res75_name = mesh_name + '_res75'
    mesh_res50_name = mesh_name + '_res50'
    mesh_res25_name = mesh_name + '_res25'
    
    # conversion en format meshlab
    gii_path = mesh_path
    subject_folder = os.path.join(data_folder, mesh_name)
    if not os.path.exists(subject_folder):
        os.makedirs(subject_folder)
    ply_path = os.path.join(subject_folder, f"{mesh_res100_name}{output_extension}")
    mp.convert_gii_to_ply(gii_path, ply_path)

    # calcul de differente resolution
    fd.apply_filter_decimation(ply_path, os.path.join(subject_folder, f"{mesh_res75_name}{output_extension}") , 0.75)
    fd.apply_filter_decimation(ply_path, os.path.join(subject_folder, f"{mesh_res50_name}{output_extension}") , 0.50)
    fd.apply_filter_decimation(ply_path, os.path.join(subject_folder, f"{mesh_res25_name}{output_extension}") , 0.25)

# boucle calcul de la dpf-star
# pour chaque sujet et resoluion  : calcule


# boucle analyse de l'histogramme