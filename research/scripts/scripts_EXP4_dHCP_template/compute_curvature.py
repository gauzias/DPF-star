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
        # compute curv
        K1, K2 = depth.curv(mesh)
        mean_curv = 0.5 * (K1 + K2)
        # save curv
        fname_K1 = 'week-' + str(week) + '_hemi-' + hemi + '_K1.gii'
        fname_K2 = 'week-' + str(week) + '_hemi-' + hemi + '_K2.gii'
        fname_mean_curv = 'week-' + str(week) + '_hemi-' + hemi + '_mean_curv.gii'
        folder_save = os.path.join(wd, 'data_EXP4/result_EXP4/depth/curvature')
        if not os.path.exists(folder_save):
            os.makedirs(folder_save)

        sio.write_texture(stex.TextureND(darray=K1), os.path.join(folder_save, fname_K1))
        sio.write_texture(stex.TextureND(darray=K2), os.path.join(folder_save, fname_K2))
        sio.write_texture(stex.TextureND(darray=mean_curv), os.path.join(folder_save, fname_mean_curv))
