import os
from research.tools.data_manager import list_files as dm 
from research.tools.mesh_processing import convert_gii_to_ply as cgp
from research.tools.mesh_processing import convert_ply_to_gii as cpg
from research.tools.mesh_processing import filter_decimation as fd
from research.tools.mesh_processing import snapshot_meshlab as snap
import research.test_sampling_robustess.path_manager as pm

# Path manager and variables
resolutions_list = pm.SAMPLING_LIST
wd = pm.WD_FOLDER
data_folder = os.path.join(wd, "data_test_sampling_robstess")
gii_extension = ".gii"
ply_extension = ".ply"

# Loop through each subject in the list
try:
    dataset_path = os.path.join(data_folder, "dataset")
    files = dm.list_files(dataset_path)
    print("\n".join(files))
except ValueError as e:
    print(e)

# Decimation loop
for mesh_path in files:
    # Get filename without extension
    mesh_name = dm.get_filename_without_extension(mesh_path, gii_extension)
    mesh_res100_name = mesh_name + "_res100"

    # Convert to MeshLab format
    subject_folder = os.path.join(data_folder, mesh_name)
    os.makedirs(subject_folder, exist_ok=True)
    path_mesh_ply = os.path.join(subject_folder, f"{mesh_res100_name}{ply_extension}")
    path_mesh_gii = os.path.join(subject_folder, f"{mesh_res100_name}{gii_extension}")
    cgp.convert_gii_to_ply(mesh_path, path_mesh_ply)
    cpg.convert_ply_to_gii(path_mesh_ply, path_mesh_gii)

    # Snapshot for resolution 100
    snapshot_folder = os.path.join(subject_folder, "snapshot")
    os.makedirs(snapshot_folder, exist_ok=True)
    output_image = os.path.join(snapshot_folder, f"{mesh_res100_name}.png")
    snap.snapshot(path_mesh_ply, output_image)

    # Compute different resolutions
    for res in resolutions_list:
        name_mesh_decimate = mesh_name + "_res" + str(res)
        path_mesh_decimate_ply = os.path.join(subject_folder, f"{name_mesh_decimate}{ply_extension}")
        path_mesh_decimate_gii = os.path.join(subject_folder, f"{name_mesh_decimate}{gii_extension}")
        fd.apply_filter_decimation(path_mesh_ply, path_mesh_decimate_ply, res / 100)
        cpg.convert_ply_to_gii(path_mesh_decimate_ply, path_mesh_decimate_gii)

        # Snapshot for decimated resolution
        output_image = os.path.join(snapshot_folder, f"{name_mesh_decimate}.png")
        snap.snapshot(path_mesh_decimate_ply, output_image)
