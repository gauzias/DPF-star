import os
import pandas as pd
import numpy as np
import slam.io as sio
import slam.texture as stex
import tools.voronoi as tv

"""
author : maxime.dieudonne@univ-amu.fr
date : 26/07/2023

this script compute the voronoi area for each mesh in the dataset 1

"""

# paths
wd = '/home/maxime/callisto/repo/paper_sulcal_depth'
dHCP_rel3_info_path = '/media/maxime/Expansion/rel3_dHCP/all_seesion_info_order.csv'
KKI_info_path = '/media/maxime/Expansion/FS_database_KKI_test_retest_FS6.0/KKI_info.csv'
folder_KKI_meshes = '/media/maxime/Expansion/FS_database_KKI_test_retest_FS6.0/'

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
session_dHCP = dHCP_rel3_info['session_id'].values
session_dHCP = [str(xx) for xx in session_dHCP]

sub_dd = list()
for file in os.listdir('/home/maxime/callisto/repo/paper_sulcal_depth/data/EXP_3_bassins/voronoi_textures'):
    if not file.endswith('_MR1_voronoi.gii'):
        subd = file.split('_')[0]
        sub_dd.append(subd)


#bugs_meshes = list()
#['CC00689XX22', 'CC00770XX12', 'CC01207XX11']
#['CC00689XX22', 'CC00770XX12', 'CC00475XX14', 'CC01207XX11']

# loop over dHCP subject
for idx, sub in enumerate(sub_dHCP) :
    print(sub, idx ,'/', nbh)
    ses = session_dHCP[idx]
    # load mesh
    mesh_name  = 'sub-' + sub + '_ses-' + ses + '_hemi-left_wm.surf.gii'
    mesh_folder = os.path.join('/media/maxime/Expansion/rel3_dHCP/', 'sub-' + sub, 'ses-' + ses,
                               'anat', )
    mesh_path = os.path.join(mesh_folder, mesh_name)

    mesh = sio.load_mesh(mesh_path)
    voronoi = tv.voronoi_de_papa(mesh)
    sio.write_texture(stex.TextureND(darray=voronoi),
                          os.path.join(wd, 'data/EXP_3_bassins/voronoi_textures', sub + '_' + ses + '_voronoi.gii'))



# loop over KKI subject
for idx, sub in enumerate(sub_KKI) :
    ses = 'MR1'
    print(sub, idx ,'/', nbk)
    # load mesh
    mesh_name  = 'lh.white.gii'
    mesh_folder = os.path.join(folder_KKI_meshes, 'KKI2009_' + sub + '_MR1', 'surf')
    mesh_path = os.path.join(mesh_folder, mesh_name)

    mesh = sio.load_mesh(mesh_path)
    voronoi = tv.voronoi_de_papa(mesh)
    sio.write_texture(stex.TextureND(darray=voronoi),
                          os.path.join(wd, 'data/EXP_3_bassins/voronoi_textures', 'KKI2009_' + sub + '_MR1' + '_voronoi.gii'))

