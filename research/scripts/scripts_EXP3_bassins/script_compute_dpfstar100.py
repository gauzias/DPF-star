import os
import pandas as pd
import numpy as np
import slam.io as sio
import slam.texture as stex
import tools.voronoi as tv
import tools.depth as depth
"""
author : maxime.dieudonne@univ-amu.fr
date : 27/07/2023

this script compute the curvature on the dataset 3 for experience 3 bassins detections

"""

# paths
wd = '/home/maxime/callisto/repo/paper_sulcal_depth'
dHCP_rel3_info_path = '/media/maxime/Expansion/rel3_dHCP/all_seesion_info_order.csv'
KKI_info_path = '/media/maxime/Expansion/FS_database_KKI_test_retest_FS6.0/KKI_info.csv'
folder_KKI_meshes = '/media/maxime/Expansion/FS_database_KKI_test_retest_FS6.0/'


alpha = 500
alpha_str = '500'

# load KKI subjects
KKI_info = pd.read_csv(KKI_info_path)
sub_KKI = KKI_info['Subject_ID'].values
sub_KKI = [str(xx) for xx in sub_KKI]
nbk = len(sub_KKI)

# load dHCP subjects
dHCP_rel3_info = pd.read_csv(dHCP_rel3_info_path)
sub_dHCP = dHCP_rel3_info['suj'].values
sub_dHCP = [xx.split('-')[1] for xx in sub_dHCP]
nbh = len(sub_dHCP)
scan_age = dHCP_rel3_info['scan_age'].values
tri_age = np.argsort(scan_age)

session_dHCP = dHCP_rel3_info['session_id'].values
session_dHCP = [str(xx) for xx in session_dHCP]

sub_dHCP= np.array(sub_dHCP)[tri_age]
session_dHCP = np.array(session_dHCP)[tri_age]


# loop over dHCP
print(".....COMPUTE DPFSTAR 100.....")
for idx, sub in enumerate(sub_dHCP):
    print(sub)
    print(idx,'/' ,nbh)
    #load mesh
    ses = session_dHCP[idx]
    mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-left_wm.surf.gii'
    mesh_folder = os.path.join('/media/maxime/Expansion/rel3_dHCP/', 'sub-' + sub, 'ses-' + ses,
                               'anat', )
    mesh_path = os.path.join(mesh_folder, mesh_name)
    mesh = sio.load_mesh(mesh_path)
    # check if already exist and compute
    fname_K1 = sub + '_' + ses + '_K1.gii'
    fname_K2 = sub + '_' + ses + '_K2.gii'
    folder_curv = os.path.join(wd, 'data/EXP_3_bassins/curvature')

    fname_dpfstar = sub + '_' + ses + '_dpfstar'+ alpha_str + '.gii'
    folder_dpfstar = os.path.join(wd, 'data/EXP_3_bassins/dpfstar' + alpha_str)
    if not os.path.exists(os.path.join(folder_dpfstar, fname_dpfstar)):
        if os.path.exists(os.path.join(folder_curv, fname_K1)):
            # load curv
            K1 = sio.load_texture(os.path.join(folder_curv, fname_K1)).darray[0]
            K2 = sio.load_texture(os.path.join(folder_curv, fname_K2)).darray[0]
            curv = 0.5 * (K1 + K2)
            # set alphas
            alphas_dpfstar = [alpha]
            # compute dpfstar
            dpfstar = depth.dpfstar(mesh, curv, alphas_dpfstar)
            # save dpfstar
            sio.write_texture(stex.TextureND(darray=dpfstar), os.path.join(folder_dpfstar, fname_dpfstar))


