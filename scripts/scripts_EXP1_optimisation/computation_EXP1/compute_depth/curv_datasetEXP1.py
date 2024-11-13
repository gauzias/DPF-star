import slam.io as sio
import os
import pandas as pd
import slam.texture as stex
import tools.depth as depth

"""
date : 10/08/2023
author : maxime.dieudonne@univ-amu.fr

this script compute the curvature with the slam package for all the subject of the dataset EXP1
"""

# wd
wd = '/home/maxime/callisto/repo/paper_sulcal_depth/'

# load datasetEXP1
datasetEXP1 = pd.read_csv(os.path.join(wd, 'datasets/dataset_EXP1.csv'))

### COMPUTE CURV

# load subjects
subjects = datasetEXP1['participant_id'].values
sessions = datasetEXP1['session_id'].values
dataset = datasetEXP1['dataset'].values
# loop
print(".....COMPUTE CURV.....")
for idx, sub in enumerate(subjects):
    print(sub)
    ses = sessions[idx]
    dset = dataset[idx]
    #load mesh
    mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-L_space-T2w_wm.surf.gii'
    mesh_path = os.path.join(wd , 'data_EXP1/meshes', mesh_name)
    mesh = sio.load_mesh(mesh_path)
    #compute curv
    K1,K2 = depth.curv(mesh)
    #save curv
    fname_K1 = sub + '_' + ses + '_K1.gii'
    fname_K2 = sub + '_' + ses + '_K2.gii'
    folder_save = os.path.join(wd, 'data_EXP1/result_EXP1/depth', sub + '_' + ses, 'curvature')
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)
    sio.write_texture(stex.TextureND(darray=K1), os.path.join(folder_save, fname_K1))
    sio.write_texture(stex.TextureND(darray=K2), os.path.join(folder_save, fname_K2))
