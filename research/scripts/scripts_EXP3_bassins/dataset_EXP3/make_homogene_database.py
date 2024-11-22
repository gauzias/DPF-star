import pandas as pd
import numpy as np
import os
import slam.io as sio
import matplotlib.pyplot as plt
import glob

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

### OAS1
# wd
OAS1_path = '/media/maxime/Expansion/FS_OASIS'

# get list of OAS1 subject
aa = glob.glob(os.path.join(OAS1_path, 'OAS1_0*'))
bb = [a for a in aa if os.path.isdir(a)]
subjects_OAS1 = [b.split(OAS1_path + '/')[1] for b in bb]

# loop
volumes_OAS1 = list()

for idx, sub in enumerate(subjects_OAS1):
    print(idx, '/', len(subjects))
    mesh_path = os.path.join(OAS1_path, sub, 'surf', 'lh.white.gii')
    mesh = sio.load_mesh(mesh_path)
    volumes_OAS1.append(mesh.volume)


### list complete KKI + rel3 + OAS1

list_volume_both = np.hstack([list_volume, list_volume_KKI])
list_subject_both = np.hstack([subjects2, subK])
list_session_both = np.hstack([sessions2, np.repeat('MR1', len(subK))])

"""

list_volume_both = np.hstack([list_volume, volumes_OAS1])
list_subject_both = np.hstack([subjects2, subjects_OAS1])

fig, ax = plt.subplots()
ax.hist(list_volume, bins = 'auto', alpha =0.5)
ax.hist(list_volume_KKI, bins = 'auto',alpha =0.5)
ax.hist(volumes_OAS1, bins = 'auto',alpha =0.5)
plt.show()
"""

## Volume classe


nb_classe = 20
nbs = 10

cc = np.linspace(np.min(list_volume_both), np.max(list_volume_both), nb_classe)
list_classe = [[cc[i], cc[i+1]] for i in np.arange(0, len(cc)-1)]

l_subject = list()
l_session = list()
l_volume = list()

for classe in list_classe:
    print(classe)
    subdf =np.where( (list_volume_both>=classe[0]) & (list_volume_both<classe[1]) )[0]
    if len(subdf)<=nbs:
        l_subject = np.hstack([ l_subject, list_subject_both[subdf] ])
        l_session = np.hstack([ l_session, list_session_both[subdf] ])
        l_volume = np.hstack( [ l_volume, list_volume_both[subdf] ] )
    if len(subdf)>nbs:
        l_subject = np.hstack([ l_subject, list_subject_both[subdf][0:nbs] ])
        l_session = np.hstack([ l_session, list_session_both[subdf][0:nbs]  ])
        l_volume = np.hstack( [ l_volume, list_volume_both[subdf][0:nbs]  ] )


fig, ax = plt.subplots()
ax.hist(list_volume_both, nb_classe)

fig2, ax2 = plt.subplots()
ax2.hist(l_volume, bins=nb_classe)

plt.show()