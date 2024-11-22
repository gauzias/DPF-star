import slam.io as sio
import os
import pandas as pd
import pickle

"""
date : 10/08/2023
author : maxime.dieudonne@univ-amu.fr

this script compute the gradient of the dmap crest with the slam package for all the subject of the dataset EXP1
"""

# wd
wd = '/home/maxime/callisto/repo/paper_sulcal_depth/'

# load datasetEXP1
datasetEXP1 = pd.read_csv(os.path.join(wd, 'datasets/dataset_EXP1.csv'))

subjects = datasetEXP1['participant_id'].values
sessions = datasetEXP1['session_id'].values
dataset = datasetEXP1['dataset'].values

### GRADIENT DPF003
print('... COMPUTE GRADIENT DMAP CREST')
for idx, sub in enumerate(subjects):
    print(sub)
    ses = sessions[idx]
    dset = dataset[idx]
    # load mesh
    #load mesh
    mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-L_space-T2w_wm.surf.gii'
    mesh_path = os.path.join(wd , 'data_EXP1/meshes', mesh_name)
    mesh = sio.load_mesh(mesh_path)
    # sulcal wall line

    # load dmap crest
    dmap_path = os.path.join(wd, 'data_EXP1/result_EXP1/depth', sub + '_' + ses, 'dmap_crest',
                               sub + '_' + ses + '_dmap_crest.gii')
    dmap = sio.load_texture(dmap_path).darray[0]
    # load gradient dmap_crest
    with open(os.path.join(wd, 'data_EXP1/sulcalWall_lines', sub + '_' + ses + '_sulcalwall.pkl'), 'rb') as file:
        sw_lines = pickle.load(file)

    # import gradient of the Surface
    grad_surface = pd.read_csv(os.path.join(wd,'data_EXP1/result_EXP1/depth',
                                            sub + '_'+ ses, 'dmap_crest/derivative', sub + '_' + ses + '_grad_dmap_crest.csv'))

    g0 = grad_surface[sw_lines]

