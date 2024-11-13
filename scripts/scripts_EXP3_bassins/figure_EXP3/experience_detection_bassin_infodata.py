
import numpy as np
import slam.io as sio
import pandas as pd
import tools.voronoi as tv
import matplotlib.pyplot as plt
import slam.texture as stex
import slam.curvature as scurv
import tools.depth as depth
import os
import slam.texture as stex
import seaborn as sns
import scipy.stats as ss





wd = '/home/maxime/callisto/repo/paper_sulcal_depth'
"""
dir_sulcKKI = '/home/maxime/callisto/data/outputs_mirtkdeformmesh_KKIFS6'
dir_dHCP = '/media/maxime/DATA/rel3_dhcp_anat_pipeline'
dir_KKI = '/home/maxime/callisto/data/databases_copied_from_hpc/REPRO_database/FS_database_KKI_test_retest_FS6.0'

dir_voronoi = '/home/maxime/callisto/repo/paper_sulcal_depth/data/rel3/voronoi'
dir_dpf100 = '/home/maxime/callisto/repo/paper_sulcal_depth/data/rel3/dpf100'


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

subjects = np.hstack([sub_dHCP, sub_KKI])
sessions = np.hstack([ses_dHCP, ses_KKI])

dataset = np.hstack([np.repeat('dHCP', len(sub_dHCP)), np.repeat('KKI', len(sub_KKI))])

#init
list_surface = np.array([])
list_vhull = np.array([])
list_volume = np.array([])
list_GI = np.array([])


df = pd.DataFrame(dict(dataset = [], subject = [], session = [], surface = [], vhull =[], volume = [], GI =[] ))

for idx, sub in enumerate(subjects):
    print(sub)
    ses = sessions[idx]
    dset = dataset[idx]
    if dset == 'dHCP':
        # load mesh
        mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-left_wm.surf.gii'
        mesh_path = os.path.join(dir_dHCP, 'sub-' + sub, 'ses-' + ses, 'anat',   mesh_name)
        mesh = sio.load_mesh(mesh_path)
    else:
        # load mesh
        mesh_path = os.path.join(dir_KKI, sub + '_' + ses, 'surf/lh.white.gii')
        mesh = sio.load_mesh(mesh_path)

    # compute features
    surface = mesh.area
    vhull = mesh.convex_hull.volume
    volume = mesh.volume
    GI = mesh.area / mesh.convex_hull.area
    # stock features
    list_surface = np.hstack([list_surface, surface])
    list_vhull = np.hstack([list_vhull, vhull])
    list_volume = np.hstack([list_volume, volume])
    list_GI = np.hstack([list_GI, GI])


df = pd.DataFrame(dict(dataset = dataset,
                            subject = subjects,
                             session = sessions,
                             surface = list_surface,
                             vhull =list_vhull,
                             volume = list_volume,
                             GI =list_GI))

df.to_csv(os.path.join(wd, 'data/info_database', 'rel3_info.csv'), index=False)

"""
df = pd.read_csv(os.path.join(wd, '../../data/info_database', 'rel3_info.csv'))

fig, axs = plt.subplots(2,2)

f1 = sns.histplot(data=df, x= "surface", ax = axs[0,0], hue = "dataset")
f2 = sns.histplot(data=df, x= "volume", ax = axs[0,1], hue = "dataset")
f3 = sns.histplot(data=df, x= "vhull", ax = axs[1,0], hue = "dataset")
f4 = sns.histplot(data=df, x= "GI", ax = axs[1,1], hue = "dataset",multiple="stack")


axs[0,0].set_title("histogram surface mm2")
axs[0,1].set_title("histogram volume mm3")
axs[1,0].set_title("histogram volume hull mm3")
axs[1,1].set_title("cumulativ histogram GI")
plt.tight_layout()
plt.show()