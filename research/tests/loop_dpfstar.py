
import os
from research.tools.data_manager import list_files as dm 
from app import compute_dpfstar



# path manager and variables
resolutions_list = [10]
data_folder  = "D:/Callisto/data/data_repo_dpfstar/data_test_resolution"
gii_extension = ".gii"

# boucle pour chaque sujet dans la liste : 
try:
    files = dm.list_files(os.path.join(data_folder, "subdataset"))
    print("\n".join(files))
except ValueError as e:
    print(e)

### Loop dcompute dpfstar

for mesh_path in files :
    print(mesh_path)
    # get name file
    mesh_name = dm.get_filename_without_extension(mesh_path, gii_extension)
    subject_folder = os.path.join(data_folder, mesh_name)

    # calcul DPF-star
    for res in resolutions_list:
        name_mesh_decimate = mesh_name + '_res' + str(res)
        path_mesh_decimate_gii = os.path.join(subject_folder, f"{name_mesh_decimate}{gii_extension}")
        compute_dpfstar.main(path_mesh_decimate_gii, display=False)


    