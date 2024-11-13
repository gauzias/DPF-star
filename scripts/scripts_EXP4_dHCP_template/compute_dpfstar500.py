import os
import pandas as pd
import numpy as np
import slam.io as sio
import slam.texture as stex
import tools.depth as depth

"""
date : 6/09/23
author : maxime.dieudonne@univ-amu.fr

this script compute the dpf star with alpha = 500 on the week template from dHCP
"""


# work directory
wd = '/home/maxime/callisto/repo/paper_sulcal_depth'
weeks = np.arange(28,45)
hemis = ['left', 'right']

for week in weeks:
    print('week:', week)
    for hemi in hemis :
        print('hemi:', hemi )
        # load mesh
        mesh_name = 'week-' + str(week) + '_hemi-' + hemi + '_space-dhcpSym_dens-32k_wm.surf.gii'
        mesh_folder = os.path.join(wd, 'data/dhcp_template/dhcpSym_template')
        mesh_path = os.path.join(mesh_folder, mesh_name)
        mesh = sio.load_mesh(mesh_path)
        # load curv
        fname_mean_curv = 'week-' + str(week) + '_hemi-' + hemi + '_mean_curv.gii'
        mean_curv_path = os.path.join(wd, 'data_EXP4/result_EXP4/depth/curvature', fname_mean_curv)
        mean_curv = sio.load_texture(mean_curv_path).darray[0]
        # compute dpfstar
        dpfstar500 = depth.dpfstar(mesh, mean_curv, alphas=[500])

        folder_save = os.path.join(wd, 'data_EXP4/result_EXP4/depth/dpfstar500')
        fname_dpfstar500 = 'week-' + str(week) + '_hemi-' + hemi + '_dpfstar500.gii'
        if not os.path.exists(folder_save):
            os.makedirs(folder_save)

        sio.write_texture(stex.TextureND(darray=dpfstar500), os.path.join(folder_save, fname_dpfstar500))

