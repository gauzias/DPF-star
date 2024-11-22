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
    else :
        subH = file.split('_')[0]
        sesH = file.split('_')[1]
        sub_dHCP.append(subH)
        ses_dHCP.append(sesH)

subjects = np.hstack([sub_dHCP, sub_KKI])
sessions = np.hstack([ses_dHCP, ses_KKI])

dataset = np.hstack([np.repeat('dHCP', len(sub_dHCP)), np.repeat('KKI', len(sub_KKI))])



fig, ax = plt.subplots()


for Th in np.arange(-0.1, 0.1, 0.02):
    list_metric_sulc = list()
    list_metric_dpf100 = list()
    list_surface = list()

    for idx, sub in enumerate(subjects) :
        print(sub)
        ses = sessions[idx]
        dset = dataset[idx]

    # load mesh
    #mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-left_wm.surf.gii'
    #mesh_path = os.path.join(dir_dHCP, 'sub-' + sub, 'ses-' + ses, 'anat',   mesh_name)
    #mesh = sio.load_mesh(mesh_path)
    #mesh_path = os.path.join(dir_KKI, sub + '_' + ses, 'surf/lh.white.gii')
    #mesh = sio.load_mesh(mesh_path)

    """
    
    #compute curv
    PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = scurv.curvatures_and_derivatives(mesh)
    K1 = PrincipalCurvatures[0, :]
    K2 = PrincipalCurvatures[1, :]
    fname_K1 = sub + '_' + ses + '_K1.gii'
    fname_K2 = sub + '_' + ses + '_K2.gii'
    folder_curv = os.path.join(wd, 'data/rel3/curvature')
    if not os.path.exists(folder_curv):
        os.makedirs(folder_curv)
    sio.write_texture(stex.TextureND(darray=K1), os.path.join(folder_curv, fname_K1))
    sio.write_texture(stex.TextureND(darray=K2), os.path.join(folder_curv, fname_K2))
    # compute dpf star 100
    K1_path = os.path.join(wd, 'data/rel3/curvature',  sub + '_' + ses + '_K1.gii')
    K2_path = os.path.join(wd, 'data/rel3/curvature',  sub + '_' + ses + '_K2.gii')
    K1 = sio.load_texture(K1_path).darray[0]
    K2 = sio.load_texture(K2_path).darray[0]
    curv = 0.5 * (K1 + K2)
    dpf100 = depth.dpfstar(mesh, curv, [100])
    dpf100 = dpf100[0]
    folder_dpf100 = os.path.join(wd, 'data/rel3/dpf100')
    if not os.path.exists(folder_dpf100):
        os.makedirs(folder_dpf100)
    name_dpf100 = sub + '_' + ses + '_dpf100.gii'
    sio.write_texture(stex.TextureND(darray=dpf100), os.path.join(folder_dpf100, name_dpf100))
    """


        if dset =='dHCP':
            # load sulc dHCP
            sulc_name = 'sub-' + sub + '_ses-' + ses + '_hemi-left_sulc.shape.gii'
            sulc_path = os.path.join(dir_dHCP, 'sub-' + sub, 'ses-' + ses, 'anat',   sulc_name)
            sulc = sio.load_texture(sulc_path).darray[0]
        if dset == 'KKI':
            # load sulc KKI
            sulc_name = sub + '_' + ses + '_l_sulc.gii'
            sulc = sio.load_texture(os.path.join(dir_sulcKKI, sulc_name)).darray[0]


        #load dpf100
        dpf100_path = os.path.join(dir_dpf100, sub + '_' + ses + '_dpf100.gii')
        dpf100 = sio.load_texture(dpf100_path).darray[0]

        # load voronoi
        voronoi_path = os.path.join(dir_voronoi, sub + '_' + ses + '_voronoi.gii')
        voronoi = sio.load_texture(voronoi_path).darray[0]

        # bassin detection
        bassin_sulc = sulc <= - 2
        bassin_dpf100 = dpf100 <= Th
        # metric
        metric_sulc = np.sum(voronoi[bassin_sulc]) / np.sum(voronoi)
        metric_dpf100 = np.sum(voronoi[bassin_dpf100]) / np.sum(voronoi)
        list_metric_sulc.append(metric_sulc)
        list_metric_dpf100.append(metric_dpf100)

        list_surface.append(np.sum(voronoi))


    #ax.scatter(list_surface, list_metric_sulc, label='sulc')
    ax.scatter(list_surface, list_metric_dpf100, label='dpf100')







ax.set_xlabel('surface area')
ax.set_ylabel('ratio surface bassin / total surface ')

plt.legend()
plt.show()

