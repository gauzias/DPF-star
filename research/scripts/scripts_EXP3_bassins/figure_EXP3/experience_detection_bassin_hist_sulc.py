
import numpy as np
import slam.io as sio
import matplotlib.pyplot as plt
import os
import seaborn as sns
import scipy.stats as ss


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

subjects = np.hstack([sub_dHCP, sub_KKI])
sessions = np.hstack([ses_dHCP, ses_KKI])
dataset = np.hstack([np.repeat('dHCP', len(sub_dHCP)), np.repeat('KKI', len(sub_KKI))])


## define threshold
value_dpf100 = np.array([])
value_sulc = np.array([])
value_sulc_Z = np.array([])
value_sulc_N = np.array([])


value2_dpf100 = list()
value2_sulc = list()
value2_sulc_Z = list()
value2_sulc_N = list()




for idx, sub in enumerate(subjects):
    print(sub)
    ses = sessions[idx]
    dset = dataset[idx]
    if dset == 'dHCP':
        # load mesh
        mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-left_wm.surf.gii'
        mesh_path = os.path.join(dir_dHCP, 'sub-' + sub, 'ses-' + ses, 'anat',   mesh_name)
        mesh = sio.load_mesh(mesh_path)
        surface = mesh.area

        # load sulc dHCP
        sulc_name = 'sub-' + sub + '_ses-' + ses + '_hemi-left_sulc.shape.gii'
        sulc_path = os.path.join(dir_dHCP, 'sub-' + sub, 'ses-' + ses, 'anat', sulc_name)

        # compute sulcs
        sulc = sio.load_texture(sulc_path).darray[0]
        sulc_N = sulc / np.power(surface, 1 / 2)
        sulc_Z = ss.zscore(sulc)

        #stock sulcs
        value_sulc = np.hstack([value_sulc, sulc])
        value_sulc_N = np.hstack([value_sulc_N, sulc_N])
        value_sulc_Z = np.hstack([value_sulc_Z, sulc_Z])

        value2_sulc.append(sulc)
        value2_sulc_N.append(sulc_N)
        value2_sulc_Z.append(sulc_Z)
    else:
        # load mesh
        mesh_path = os.path.join(dir_KKI, sub + '_' + ses, 'surf/lh.white.gii')
        mesh = sio.load_mesh(mesh_path)
        surface = mesh.area
        # load sulc KKI
        sulc_name = sub + '_' + ses + '_l_sulc.gii'

        # compute sulcs
        sulc = sio.load_texture(os.path.join(dir_sulcKKI, sulc_name)).darray[0]
        sulc_N = sulc / np.power(surface, 1 / 2)
        sulc_Z = ss.zscore(sulc)
        # stock sulcs
        value_sulc = np.hstack([value_sulc, sulc])
        value_sulc_N = np.hstack([value_sulc_N, sulc_N])
        value_sulc_Z = np.hstack([value_sulc_Z, sulc_Z])

        value2_sulc.append(sulc)
        value2_sulc_N.append(sulc_N)
        value2_sulc_Z.append(sulc_Z)

    # load dpf100
    dpf100_path = os.path.join(dir_dpf100, sub + '_' + ses + '_dpf100.gii')
    dpf100 = sio.load_texture(dpf100_path).darray[0]
    dpf100 = 20*dpf100
    value_dpf100 = np.hstack([value_dpf100, dpf100])

    value2_dpf100.append(dpf100)




fig, axs = plt.subplots(4)

f1 = sns.histplot(value_sulc, ax = axs[0])
f2 = sns.histplot(value_sulc_N, ax = axs[1])
f3 = sns.histplot(value_sulc_Z, ax = axs[2])
f4 = sns.histplot(value_dpf100, ax = axs[3])

axs[0].set_title("sulc")
axs[1].set_title("sulc Zscore-normalised")
axs[2].set_title("sulc size-noramlised")
axs[3].set_title("DPF-star alpha100")


fig2, ax2s = plt.subplots(4)

ax2s[0].boxplot(value2_sulc)
ax2s[1].boxplot(value2_sulc_N)
ax2s[2].boxplot(value2_sulc_Z)
ax2s[3].boxplot(value2_dpf100)

ax2s[0].set_title("sulc")
ax2s[1].set_title("sulc Zscore-normalised")
ax2s[2].set_title("sulc size-normalised")
ax2s[3].set_title("DPF-star alpha100")

plt.tight_layout()
plt.show()