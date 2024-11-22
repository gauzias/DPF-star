import slam.io as sio
import os
import pandas as pd
import tools.gradient as gradient

"""
date : 10/08/2023
author : maxime.dieudonne@univ-amu.fr

this script compute the gradient of the dpf 0.03 with the slam package for all the subject of the dataset EXP1
"""

# wd
wd = '/home/maxime/callisto/repo/paper_sulcal_depth/'

# load datasetEXP1
datasetEXP1 = pd.read_csv(os.path.join(wd, 'datasets/dataset_EXP1.csv'))

subjects = datasetEXP1['participant_id'].values
sessions = datasetEXP1['session_id'].values
dataset = datasetEXP1['dataset'].values

### GRADIENT DPF003
print('... COMPUTE GRADIENT DPF 0.03')
for idx, sub in enumerate(subjects):
    print(sub)
    ses = sessions[idx]
    dset = dataset[idx]
    # load mesh
    #load mesh
    mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-L_space-T2w_wm.surf.gii'
    mesh_path = os.path.join(wd , 'data_EXP1/meshes', mesh_name)
    mesh = sio.load_mesh(mesh_path)
    # load dpf003
    dpf003_path = os.path.join(wd, 'data_EXP1/result_EXP1/depth', sub + '_' + ses, 'dpf003',
                               sub + '_' + ses + '_dpf003.gii')
    dpf003 = sio.load_texture(dpf003_path).darray[0]
    # compute gradient
    grad_dpf003 = gradient.gradient_texture(dpf003, mesh)
    df_grad_dpf003 = pd.DataFrame(grad_dpf003, columns=['x', 'y', 'z'])
    folder_grad_dpf003 = os.path.join(wd, 'data_EXP1/result_EXP1/depth', sub + '_' + ses, 'dpf003', 'derivative')
    if not os.path.exists(folder_grad_dpf003):
        os.makedirs(folder_grad_dpf003)
    df_grad_dpf003.to_csv(os.path.join(folder_grad_dpf003, sub + '_' + ses + '_grad_dpf003.csv'), index=False)

