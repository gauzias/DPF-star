import slam.io as sio
import os
import pandas as pd
import tools.gradient as gradient

"""
date : 10/08/2023
author : maxime.dieudonne@univ-amu.fr

this script compute the gradient of sulc with the slam package for all the subject of the dataset EXP1
"""

# wd
wd = '/home/maxime/callisto/repo/paper_sulcal_depth/'

# load datasetEXP1
datasetEXP1 = pd.read_csv(os.path.join(wd, 'datasets/dataset_EXP1.csv'))

subjects = datasetEXP1['participant_id'].values
sessions = datasetEXP1['session_id'].values
dataset = datasetEXP1['dataset'].values

### GRADIENT DPF003
print('... COMPUTE GRADIENT SULC')
for idx, sub in enumerate(subjects):
    print(sub)
    ses = sessions[idx]
    dset = dataset[idx]
    # load mesh
    mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-L_space-T2w_wm.surf.gii'
    mesh_path = os.path.join(wd , 'data_EXP1/meshes', mesh_name)
    mesh = sio.load_mesh(mesh_path)
    # load dpf003
    if dset == 'dHCP':
        sulc_name = 'sub-' + sub + '_ses-'+ ses+'_hemi-L_space-T2w_sulc.shape.gii'
    if dset == 'KKI2009':
        sulc_name = sub + '_' + ses + '_sulc.gii'
    sulc_path = os.path.join(wd, 'data_EXP1/result_EXP1/depth', sub + '_' + ses, 'sulc', sulc_name)
    sulc = sio.load_texture(sulc_path).darray[0]
    # compute gradient
    grad_sulc = gradient.gradient_texture(sulc, mesh)
    df_grad_sulc = pd.DataFrame(grad_sulc, columns=['x', 'y', 'z'])
    folder_grad_sulc = os.path.join(wd, 'data_EXP1/result_EXP1/depth', sub + '_' + ses, 'sulc', 'derivative')
    if not os.path.exists(folder_grad_sulc):
        os.makedirs(folder_grad_sulc)
    df_grad_sulc.to_csv(os.path.join(folder_grad_sulc, sub + '_' + ses + '_grad_sulc.csv'), index=False)

