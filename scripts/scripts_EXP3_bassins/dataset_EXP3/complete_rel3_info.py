import pandas as pd
import numpy as np
import slam.io as sio
import os
import scipy.stats as ss
import pickle
import locale
locale.atof('123,456')

# paths
wd = '/home/maxime/callisto/repo/paper_sulcal_depth'
folder_dHCP = '/media/maxime/Expansion/rel3_dHCP'
info_data_path = os.path.join(wd, 'data/info_database/info_database_EXP3.csv')
df = pd.read_csv(info_data_path)

subjects = df['suj'].values
subjects = [xx.split('-')[1] for xx in subjects]
nbh = len(subjects)
sessions = df['session_id'].values
scan_age = df['scan_age_days'].values
dataset = df['dataset'].values


"""
for idx, sub in enumerate(subjects):
    ses = sessions[idx]
    dset = dataset[idx]
    # check if curvature is computed
    if (dset == 'dHCP'):
        mesh_path = os.path.join('/media/maxime/Expansion/rel3_dHCP', 'sub-' + sub, 'ses-' + ses,
                             'anat', 'sub-' + sub + '_ses-' + ses + '_hemi-left_wm.surf.gii')
        mesh = sio.load_mesh(mesh_path)
    if (dset == 'KKI2009'):
        print('yeah  ! ')
        # load mesh
        mesh_path = os.path.join('/media/maxime/Expansion/FS_database_KKI_test_retest_FS6.0',
                                     'KKI2009_' + sub + '_' + ses, 'surf', 'lh.white.gii')
        mesh = sio.load_mesh(mesh_path)

    # compute surface, volume, GI
    surface = mesh.area
    volume = mesh.volume
    GI = mesh.area / mesh.convex_hull.area
    # store in df
    index = df.index[(df['suj'] == 'sub-' + sub) & (df['session_id']==ses)].tolist()
    print(index)
    df.loc[index, 'GI'] = GI
    df.loc[index, 'volume'] = volume
    df.loc[index, 'surface'] = surface
    print(sub, GI)
"""
# check process steps done

for idx, sub in enumerate(subjects):
    print(sub)
    ses = sessions[idx]
    dset = dataset[idx]
    index = df.index[(df['suj'] == 'sub-' + sub) & (df['session_id'] == ses)].tolist()
    if dset == 'dHCP':
        # check curvature
        fname_K1 = sub + '_' + str(ses) + '_K1.gii'
        folder_curv = os.path.join(wd, 'data/EXP_3_bassins/curvature')
        if os.path.exists(os.path.join(folder_curv, fname_K1)):
            print('curv exist')
            df.loc[index, 'curvature'] = True
        if not os.path.exists(os.path.join(folder_curv, fname_K1)):
            print('curv not exist')
            df.loc[index, 'curvature'] = False
        # check dpf100
        fname_dpftsar100 =  sub + '_' + str(ses) + '_dpfstar100.gii'
        folder_dpfstar100 = os.path.join(wd, 'data/EXP_3_bassins/dpfstar100')
        if os.path.exists(os.path.join(folder_dpfstar100, fname_dpftsar100)):
            df.loc[index, 'dpfstar100'] = True
        if not os.path.exists(os.path.join(folder_dpfstar100, fname_dpftsar100)):
            df.loc[index, 'dpfstar100'] = False
        # check voronoi
        fname_voronoi =  sub + '_' + str(ses) + '_voronoi.gii'
        folder_voronoi = os.path.join(wd, 'data/EXP_3_bassins/voronoi_textures')
        if os.path.exists(os.path.join(folder_voronoi, fname_voronoi)):
            df.loc[index, 'voronoi'] = True
        if not os.path.exists(os.path.join(folder_voronoi, fname_voronoi)):
            df.loc[index, 'voronoi'] = False

    if dset == 'KKI2009':
        # check curvature
        fname_K1 = 'KKI2009_' + sub + '_' + str(ses) + '_K1.gii'
        folder_curv = os.path.join(wd, 'data/EXP_3_bassins/curvature')
        if os.path.exists(os.path.join(folder_curv, fname_K1)):
            df.loc[index, 'curvature'] = True
        if not os.path.exists(os.path.join(folder_curv, fname_K1)):
            df.loc[index, 'curvature'] = False
        # check dpf100
        fname_dpftsar100 = 'KKI2009_' + sub + '_' + str(ses) + '_dpfstar100.gii'
        folder_dpfstar100 = os.path.join(wd, 'data/EXP_3_bassins/dpfstar100')
        if os.path.exists(os.path.join(folder_dpfstar100, fname_dpftsar100)):
            df.loc[index, 'dpfstar100'] = True
        if not os.path.exists(os.path.join(folder_dpfstar100, fname_dpftsar100)):
            df.loc[index, 'dpfstar100'] = False
        # check voronoi
        fname_voronoi =  'KKI2009_' + sub + '_' + str(ses) + '_voronoi.gii'
        folder_voronoi = os.path.join(wd, 'data/EXP_3_bassins/voronoi_textures')
        if os.path.exists(os.path.join(folder_voronoi, fname_voronoi)):
            df.loc[index, 'voronoi'] = True
        if not os.path.exists(os.path.join(folder_voronoi, fname_voronoi)):
            df.loc[index, 'voronoi'] = False


#save
df.to_csv(os.path.join(wd, 'data/info_database/info_database_EXP3.csv'), index=False)

import seaborn as sns
import matplotlib.pyplot as plt




import locale
aa = [locale.atof(a) for a in df[(df['dataset']=='dHCP') ]['scan_age'].values]
rr = df[(df['dataset']=='dHCP') ]['radiology_score'].values


plt.plot(rr, aa, '.')
plt.show()