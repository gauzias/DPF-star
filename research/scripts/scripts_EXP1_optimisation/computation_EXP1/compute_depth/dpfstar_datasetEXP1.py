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

# alpha
alphas = [0, 1, 5, 10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 2000]
#alphas = [5000, 10000]
df_alphas = pd.DataFrame(dict(alphas = alphas))
adaptation = 'volume'
### COMPUTE DPF
print(".....COMPUTE DPF.....")
for idx, sub in enumerate(subjects):
    print(sub)
    ses = sessions[idx]
    dset = dataset[idx]
    # load mesh
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
    dpfstar = depth.dpfstar(mesh, curv, alphas, adaptation=adaptation)
    #save dpf
    fname_dpfstar = sub + '_' + ses + '_dpfstar_' + adaptation  + '.gii'
    folder_dpfstar = os.path.join('data_EXP1/result_EXP1/depth', sub + '_' + ses, 'dpfstar_' + adaptation)
    if not os.path.exists(folder_dpfstar):
        os.makedirs(folder_dpfstar)

    df_alphas.to_csv(os.path.join(folder_dpfstar, 'alpha.csv'), index=False)
    sio.write_texture(stex.TextureND(darray=dpfstar), os.path.join(folder_dpfstar, fname_dpfstar))