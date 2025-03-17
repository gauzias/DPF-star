
import os
from research.tools.data_manager import list_files as dm 
from research.tools.mesh_processing import convert_gii_to_ply as cgp
from research.tools.mesh_processing import convert_ply_to_gii as cpg
from research.tools.mesh_processing import filter_decimation as fd
from research.tools.mesh_processing import snapshot_meshlab as snap

# path manager and variables
resolutions_list = [10]
data_folder  = "D:/Callisto/data/data_repo_dpfstar/data_test_resolution"
gii_extension = ".gii"
ply_extension = ".ply"

# boucle pour chaque sujet dans la liste : 
try:
    files = dm.list_files(os.path.join(data_folder, "subdataset"))
    print("\n".join(files))
except ValueError as e:
    print(e)

### Loop decimation

for mesh_path in files :
    # get name file
    mesh_name = dm.get_filename_without_extension(mesh_path, gii_extension)
    mesh_res100_name = mesh_name + '_res100'
    
    # conversion en format meshlab
    subject_folder = os.path.join(data_folder, mesh_name)
    if not os.path.exists(subject_folder):
        os.makedirs(subject_folder)
    path_mesh_ply = os.path.join(subject_folder, f"{mesh_res100_name}{ply_extension}")
    path_mesh_gii = os.path.join(subject_folder, f"{mesh_res100_name}{gii_extension}")
    cgp.convert_gii_to_ply(mesh_path, path_mesh_ply)
    cpg.convert_ply_to_gii(path_mesh_ply, path_mesh_gii)


    # snap
    if not os.path.exists(os.path.join(subject_folder, "snapshot")):
        os.makedirs(os.path.join(subject_folder, "snapshot"))
    output_image = os.path.join(subject_folder, "snapshot", mesh_res100_name + ".png")
    snap.snapshot(path_mesh_ply, output_image)

    # calcul de differente resolution
    for res in resolutions_list:
        name_mesh_decimate = mesh_name + '_res' + str(res)
        path_mesh_decimate_ply = os.path.join(subject_folder, f"{name_mesh_decimate}{ply_extension}")
        path_mesh_decimate_gii = os.path.join(subject_folder, f"{name_mesh_decimate}{gii_extension}")
        fd.apply_filter_decimation(path_mesh_ply, path_mesh_decimate_ply , res/100)
        cpg.convert_ply_to_gii(path_mesh_decimate_ply, path_mesh_decimate_gii)

        # snapshot
        if not os.path.exists(os.path.join(subject_folder, "snapshot")):
            os.makedirs(os.path.join(subject_folder, "snapshot"))
        output_image = os.path.join(subject_folder, "snapshot", name_mesh_decimate + ".png")

        snap.snapshot(path_mesh_decimate_ply, output_image)

    