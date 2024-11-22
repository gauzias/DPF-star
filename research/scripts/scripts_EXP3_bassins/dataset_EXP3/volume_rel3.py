import pandas as pd
import numpy as np
import os
import slam.io as sio
import matplotlib.pyplot as plt

"""
this script show the distribution of volume of rel 3 + KKI
"""


### dHCP rel3

# read info csv about rel3 dHCP
df_path = '/media/maxime/Expansion/rel3_dHCP/all_seesion_info.csv'
df = pd.read_csv(df_path)

# get list of subject
subjects = df['suj'].values
sessions = df['session_id'].values
sessions = [str(ses) for ses in sessions]

# check if sulc and surf are downloaded
sulc_copied = list()
surf_copied = list()

for idx, sub in enumerate(subjects):
    print(sub)
    ses = sessions[idx]
    sulc_path = os.path.join('/media/maxime/Expansion/rel3_dHCP', sub, 'ses-' + ses, 'anat',
                             sub + '_ses-' + ses + '_hemi-left_sulc.shape.gii')
    surf_path = os.path.join( '/media/maxime/Expansion/rel3_dHCP', sub, 'ses-' + ses, 'anat',
                              sub + '_ses-' + ses + '_hemi-left_wm.surf.gii')
    if os.path.exists(sulc_path):
        sulc_copied.append(True)
    else:
        sulc_copied.append(False)
    if os.path.exists(surf_path):
        surf_copied.append(True)
    else:
        surf_copied.append(False)

df['surf_copied'] = surf_copied
df['sulc_copied'] = sulc_copied


## get distribution of volume


df2 = df[df['sulc_copied'] == True]
subjects2 = df2['suj'].values
sessions2 = df2['session_id'].values
sessions2 = [str(ses) for ses in sessions2]


list_volume = list()
for idx, sub in enumerate(subjects2):
    print(idx,'/', len(subjects2))
    ses = sessions2[idx]
    # import mesh
    surf_path = os.path.join('/media/maxime/Expansion/rel3_dHCP', sub, 'ses-' + ses, 'anat',
                             sub + '_ses-' + ses + '_hemi-left_wm.surf.gii')
    mesh = sio.load_mesh(surf_path)

    volume = mesh.volume
    list_volume.append(volume)

df2['volume'] = list_volume

### KKI

dfKKI_path = '/home/maxime/callisto/repo/paper_sulcal_depth/datasets/dataset_KKI.csv'
dfKKI = pd.read_csv(dfKKI_path)

subK = dfKKI['Subject_ID'].values

list_volume_KKI = list()
for idx, sub in enumerate(subK):
    print(sub)

    ses = 'MR1'
    # import mesh
    surf_path = os.path.join('/media/maxime/Expansion/FS_database_KKI_test_retest_FS6.0',
                             sub.split('-')[0] + '_' + sub.split('-')[1] + '_MR1', 'surf',
                                                     'lh.white.gii')

    mesh = sio.load_mesh(surf_path)

    volume = mesh.volume
    list_volume_KKI.append(volume)

### list complete KKI + rel3

list_volume_both = np.hstack([list_volume, list_volume_KKI])

fig, ax = plt.subplots()
ax.hist(list_volume, bins =6)

fig2, ax2 = plt.subplots()
ax2.hist(list_volume_KKI, bins ='auto')

fig3, ax3 = plt.subplots()
ax3.hist(list_volume_both, bins=10)


plt.show()

# 33 - 83 - 133 - 183 - 233 - 283 - 333 -  305

vv = np.linspace(np.floor(np.min(list_volume_both)), np.ceil(np.max(list_volume_both)), 10)

vv = [np.round(v) for v in vv]

c1 = [33899.0, 63958.0]
c2 = [63958.0, 94018.0]
c3 = [94018.0, 124077.0]
c4 = [124077.0, 154136.0]
c5 = [154136.0, 184196.0]
c6 = [184196.0, 214255.0]
c7 = [214255.0, 244314.0]
c9 = [244314.0, 274374.0]
c10 = [274374.0, 304433.0]


