import os
import pandas as pd
import numpy as np
import slam.io as sio

import tools.voronoi as tv
import matplotlib.pyplot as plt
import slam.texture as stex
import slam.curvature as scurv
import tools.depth as depth
import os
import slam.texture as stex
import seaborn as sns
import scipy.stats as ss
# load sulc

wd = '/home/maxime/callisto/repo/paper_sulcal_depth'
dir_sulcKKI = '/home/maxime/callisto/data/outputs_mirtkdeformmesh_KKIFS6'
dir_dHCP = '/media/maxime/DATA/rel3_dhcp_anat_pipeline'
dir_KKI = '/home/maxime/callisto/data/databases_copied_from_hpc/REPRO_database/FS_database_KKI_test_retest_FS6.0'
dir_voronoi = '/data/rel3/voronoi'
dir_dpf100 = '/data/rel3/dpf100'

sub_KKI = list()
ses_KKI = list()
sub_dHCP = list()
ses_dHCP = list()

for file in os.listdir(dir_voronoi):
    if file.endswith('_MR1_voronoi.gii'):
        subK = file.split('_')[0] + '_' + file.split('_')[1]
        sesK = 'MR1'
        sub_KKI.append(subK)
        ses_KKI.append(sesK)
    else:
        subH = file.split('_')[0]
        sesH = file.split('_')[1]
        sub_dHCP.append(subH)
        ses_dHCP.append(sesH)


subjects_add = ["CC00735XX18","CC00672AN13", "CC00672BN13", "CC00621XX11", "CC00829XX21",
            "CC00617XX15", "CC00385XX15"]

sessions_add = ["222201", "197601" , "200000", "177900", "17610",  "176500" ,"118500" ]


subjects = np.hstack([sub_dHCP, sub_KKI, subjects_add])
sessions = np.hstack([ses_dHCP, ses_KKI, sessions_add])

dataset = np.hstack([np.repeat('dHCP', len(sub_dHCP)), np.repeat('KKI', len(sub_KKI)),
                     np.repeat('dHCP', len(subjects_add))])

addi = np.hstack([np.repeat(0, len(sub_dHCP)+len(sub_KKI)), np.repeat(1,len(subjects_add))])



## define threshold
value_dpf100 = np.array([])
value_sulc = np.array([])
value_sulc_Z = np.array([])
value_sulc_N = np.array([])

GI_list = np.array([])
surface_list = np.array([])

for idx, sub in enumerate(subjects):
    print(sub)
    ses = sessions[idx]
    dset = dataset[idx]
    add = addi[idx]


    #load mesh and SULC
    if (dset == 'dHCP') & (add == 0 ):
        # load mesh
        try :
            mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-left_wm.surf.gii'
            mesh_path = os.path.join(dir_dHCP, 'sub-' + sub, 'ses-' + ses, 'anat',   mesh_name)
            mesh = sio.load_mesh(mesh_path)
            # load sulc dHCP
            sulc_name = 'sub-' + sub + '_ses-' + ses + '_hemi-left_sulc.shape.gii'
            sulc_path = os.path.join(dir_dHCP, 'sub-' + sub, 'ses-' + ses, 'anat', sulc_name)
            sulc = sio.load_texture(sulc_path).darray[0]
            # compute SULC N and Z
            surface = mesh.area
            sulc_N = sulc / np.power(surface, 1 / 2)
            sulc_Z = ss.zscore(sulc)
        except:
            print('error')

    if (dset == 'dHCP') & (add == 1):
        # load mesh
        try :
            mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-L_space-T2w_wm.surf.gii'
            mesh_path = os.path.join('/data/meshes', mesh_name)
            mesh = sio.load_mesh(mesh_path)
            # load sulc dHCP
            sulc_name = 'sub-' + sub + '_ses-' + ses + '_hemi-L_space-T2w_sulc.shape.gii'
            sulc_path = os.path.join('/data/subject_analysis',
                                     sub + '_' + ses, 'surface_processing/sulc',
                                     sulc_name)
            sulc = sio.load_texture(sulc_path).darray[0]
            # compute SULC N and Z
            surface = mesh.area
            sulc_N = sulc / np.power(surface, 1 / 2)
            sulc_Z = ss.zscore(sulc)
        except :
            print('error')
    if (dset == 'KKI'):
        # load mesh
        print('bu')
        try :
            mesh_path = os.path.join(dir_KKI, sub + '_' + ses, 'surf/lh.white.gii')
            mesh = sio.load_mesh(mesh_path)
            # load sulc KKI
            sulc_name = sub + '_' + ses + '_l_sulc.gii'
            sulc = sio.load_texture(os.path.join(dir_sulcKKI, sulc_name)).darray[0]
            # compute SULC N and Z
            surface = mesh.area
            sulc_N = sulc / np.power(surface, 1 / 2)
            sulc_Z = ss.zscore(sulc)
        except:
            print('error')



    # write texture
    sio.write_texture(stex.TextureND(darray=sulc_N), os.path.join(wd, '../../data/rel3/sulc_relative',
                                                                  sub + '_' + ses + '_area_' +'_sulcN.gii'))
    sio.write_texture(stex.TextureND(darray=sulc_Z), os.path.join(wd, '../../data/rel3/sulc_relative',
                                                                  sub + '_' + ses + '_area_' +'_sulcZ.gii'))


