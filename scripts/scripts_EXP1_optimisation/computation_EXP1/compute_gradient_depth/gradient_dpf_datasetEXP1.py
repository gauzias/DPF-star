import slam.io as sio
import os
import pandas as pd
import numpy as np
import tools.gradient as gradient

"""
date : 10/08/2023
author : maxime.dieudonne@univ-amu.fr

this script compute the gradient of the dpf with the slam package for all the subject of the dataset EXP1
"""

# wd
wd = '/home/maxime/callisto/repo/paper_sulcal_depth/'

# load datasetEXP1
datasetEXP1 = pd.read_csv(os.path.join(wd, 'datasets/dataset_EXP1.csv'))

subjects = datasetEXP1['participant_id'].values
sessions = datasetEXP1['session_id'].values
dataset = datasetEXP1['dataset'].values

### GRADIENT CURV
print('... COMPUTE GRADIENT CURV')
for idx, sub in enumerate(subjects):
    print(sub)
    ses = sessions[idx]
    dset = dataset[idx]
    # load mesh
    #load mesh
    mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-L_space-T2w_wm.surf.gii'
    mesh_path = os.path.join(wd , 'data_EXP1/meshes', mesh_name)
    mesh = sio.load_mesh(mesh_path)
    # load alpha
    alphas_path = os.path.join(wd, 'data_EXP1/result_EXP1/depth', sub + '_' + ses, 'dpf','alpha.csv')
    alphas = pd.read_csv(alphas_path)
    alphas = alphas['alphas'].values
    print(alphas)
    # load
    depth_path = os.path.join(wd, 'data_EXP1/result_EXP1/depth', sub + '_' + ses, 'dpf', sub + '_' + ses + '_dpf.gii')
    depth = sio.load_texture(depth_path).darray
    # compute gradient
    grad_depth = gradient.gradient_texture(depth, mesh)
    grad_depth = np.array(grad_depth)
    nb_alpha = grad_depth.shape[0]
    nb_vtx = grad_depth.shape[1]
    nb_dim = grad_depth.shape[2]
    grad_depth = grad_depth.reshape(nb_alpha * nb_vtx, nb_dim)
    df_grad = pd.DataFrame(grad_depth, columns=['x', 'y', 'z'])
    df_grad['alpha'] = np.repeat(alphas, nb_vtx)
    # save
    save_folder = os.path.join(wd, 'data_EXP1/result_EXP1/depth',sub + '_' + ses , 'dpf/derivative')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    df_grad.to_csv(os.path.join(save_folder, sub + '_' + ses + '_grad_dpf.csv'), index=False)

