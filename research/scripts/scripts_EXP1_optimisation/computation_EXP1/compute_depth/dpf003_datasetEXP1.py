
import slam.io as sio
import os
import pandas as pd
import slam.texture as stex
import tools.depth as depth

"""
date : 10/08/2023
author : maxime.dieudonne@univ-amu.fr

this script compute the dpf with alpha = 0.03 with the slam package for all the subject of the dataset EXP1
"""

# wd
wd = '/home/maxime/callisto/repo/paper_sulcal_depth/'

# load datasetEXP1
datasetEXP1 = pd.read_csv(os.path.join(wd, 'datasets/dataset_EXP1.csv'))

subjects = datasetEXP1['participant_id'].values
sessions = datasetEXP1['session_id'].values
dataset = datasetEXP1['dataset'].values

### COMPUTE DPF
print(".....COMPUTE DPF.....")
for idx, sub in enumerate(subjects):
    print(sub)
    ses = sessions[idx]
    dset = dataset[idx]
    #load mesh
    mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-L_space-T2w_wm.surf.gii'
    mesh_path = os.path.join(wd , 'data_EXP1/meshes', mesh_name)
    mesh = sio.load_mesh(mesh_path)
    #load curv
    K1_path = os.path.join(wd, 'data_EXP1/result_EXP1/depth', sub + '_' + ses,  'curvature',
                           sub + '_' + ses + '_K1.gii')
    K2_path = os.path.join(wd, 'data_EXP1/result_EXP1/depth', sub + '_' + ses,  'curvature',
                           sub + '_' + ses + '_K2.gii')
    K1 = sio.load_texture(K1_path).darray[0]
    K2 = sio.load_texture(K2_path).darray[0]
    curv = 0.5 * (K1 + K2)
    # compute dpf
    dpf = depth.depth_potential_function(mesh, curv, [0.03])
    #save dpf
    fname_dpf = sub + '_' + ses + '_dpf003.gii'
    folder_dpf = os.path.join('data_EXP1/result_EXP1/depth', sub + '_' + ses, 'dpf003')
    if not os.path.exists(folder_dpf):
        os.makedirs(folder_dpf)

    sio.write_texture(stex.TextureND(darray=-dpf[0]), os.path.join(folder_dpf, fname_dpf))