import os
from research.tools.data_manager import list_files as dm 
from app import compute_dpfstar
import research.test_sampling_robustess.path_manager as pm

# Path manager and variables
resolutions_list = pm.SAMPLING_LIST
wd = pm.WD_FOLDER
data_folder = os.path.join(wd, "data_test_sampling_robustess")
gii_extension = ".gii"

# Loop through each subject in the list
try:
    dataset_path = os.path.join(data_folder, "dataset")
    files = dm.list_files(dataset_path)
    print("\n".join(files))
except ValueError as e:
    print(e)

# Loop to compute DPF-star
for mesh_path in files:
    print(mesh_path)
    # Get filename without extension
    mesh_name = dm.get_filename_without_extension(mesh_path, gii_extension)
    subject_folder = os.path.join(data_folder,"sampled_meshes", mesh_name)

    # Compute DPF-star for each resolution
    for res in resolutions_list:
        name_mesh_decimate = mesh_name + "_res" + str(res)
        path_mesh_decimate_gii = os.path.join(subject_folder, f"{name_mesh_decimate}{gii_extension}")
        compute_dpfstar.main(path_mesh_decimate_gii, display=False)
