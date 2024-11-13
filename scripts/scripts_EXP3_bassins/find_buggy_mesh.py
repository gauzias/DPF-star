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

# load dHCP subjects
dHCP_rel3_info = pd.read_csv(dHCP_rel3_info_path)
sub_dHCP = dHCP_rel3_info['suj'].values
sub_dHCP = [xx.split('-')[1] for xx in sub_dHCP]
nbh = len(sub_dHCP)
session_dHCP = dHCP_rel3_info['session_id'].values
session_dHCP = [str(xx) for xx in session_dHCP]



bugs_meshes = list()

#[['CC00805XX13', '1700'],
# ['CC00605XX11', '172700'],
# ['CC00605XX11', '187700'],
# ['CC00579XX19', '173500'],
# ['CC00439XX19', '132100'],
# ['CC01103XX06', '101930']]


# loop over dHCP subject
for idx, sub in enumerate(sub_dHCP) :
    print(sub, idx ,'/', nbh)
    ses = session_dHCP[idx]
    # load mesh
    mesh_name  = 'sub-' + sub + '_ses-' + ses + '_hemi-left_wm.surf.gii'
    mesh_folder = os.path.join('/media/maxime/Expansion/rel3_dHCP/', 'sub-' + sub, 'ses-' + ses,
                               'anat', )
    mesh_path = os.path.join(mesh_folder, mesh_name)

    try :
        mesh = sio.load_mesh(mesh_path)
    except:
        bugs_meshes.append([sub, ses])



print(bugs_meshes)