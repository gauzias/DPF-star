import tools.slam_dpf as sdpf
import slam.texture as stex
import seaborn as sns
import settings.path_manager as pm
import numpy as np
import slam.io as sio
import pandas as pd
import os
import tools.io as tio
import tools.sulcaldepth_metrics as sm
import settings.global_variables as gv
import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm
cmap = matplotlib.cm.get_cmap('plasma')

from matplotlib.colors import LinearSegmentedColormap


"""
author : maxime.dieudonne@univ-amu.fr
date : 25/07/2023

This script generate the figure of the experience 1 : optimisation of the alpha for the dpf-star method.
we compute the 3 metrics : (1) standart-deviation of the depth on crests ,  (2) difference of the median between 
depth on the fundi and the crest, (3) angular deviation of the sulcal depth on the depth. 
we compare it to the sulc and dpf standart method.

The metrics are already computed.

1/ we load the metrics saved in pkl format
2/ We create a dataframe for each metrics with inputs : 
sub : subject identifiant
depth : depth method
value : value of the metric
3/ we use functionalities of seaborn using dataframe to make plot of interest
"""


# param
alphas_dpfstar = [0, 1, 5, 10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 2000]
alphas_dpf003 = [0, 0.0001, 0.002, 0.0003, 0.0004, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.5, 1]


# path manager
wd = '/home/maxime/callisto/repo/paper_sulcal_depth'
rater = 'MD_full'
folder_info_dataset = pm.folder_info_dataset
folder_meshes = pm.folder_meshes
folder_manual_labelisation = pm.folder_manual_labelisation
folder_subject_analysis = pm.folder_subject_analysis

# load data
info_database = pd.read_csv(os.path.join(folder_info_dataset, 'info_database.csv'))
list_subs = info_database['participant_id'].values[0:-3]
list_ses = info_database['session_id'].values[0:-3]
list_scan_sage = info_database['scan_age'].values[0:-3]

list_volume = list()
for idx, sub in enumerate(list_subs) :
    ses = list_ses[idx]
    mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-L_space-T2w_wm.surf.gii'
    mesh_path = os.path.join(folder_meshes, mesh_name)
    mesh = sio.load_mesh(mesh_path)
    vol = mesh.volume
    list_volume.append(vol)


# trier les sujets par volumes
argsort_volume = np.argsort(list_volume)
list_subs = list_subs[argsort_volume]
list_ses = list_ses[argsort_volume]
list_scan_sage = list_scan_sage[argsort_volume]
list_volume = np.array(list_volume)[argsort_volume]

# load
path_table_labels_line = os.path.join(pm.folder_info_dataset, 'labels_lines.csv')
df_labels_lines = tio.get_labels_lines(path_table_labels_line)
path_table_wallpinches_info = os.path.join(pm.folder_info_dataset, 'wallpinches_info.csv')
df_wp_info = tio.get_wallpinches_info(path_table_wallpinches_info)


#### import metrics DPF-star
folder_metrics = os.path.join(wd,'data','group_analysis', 'dev')
list_mean_dev_dpfs = list()
list_var_crest_dpfs = list()
list_diff_fundicrest_dpfs = list()

for idx, sub in enumerate(list_subs):
    print(sub)
    ses = list_ses[idx]
    # DEV
    with open(os.path.join(folder_metrics, sub + '_' + ses + '_angle_dev_mean.pkl'), 'rb') as file:
        mean_dev_dpfs = pickle.load(file)
    list_mean_dev_dpfs.append(mean_dev_dpfs)
    # VAR
    with open(os.path.join(folder_metrics, sub + '_' + ses + '_var_crest_dpf_norm.pkl'), 'rb') as file:
        var_crest_dpfs = pickle.load(file)
    var_crest_dpfs = [np.sqrt(vv) for vv in var_crest_dpfs]
    list_var_crest_dpfs.append(var_crest_dpfs)
    # DIFF
    with open(os.path.join(folder_metrics, sub + '_' + ses + '_diff_dunficrest_dpf_norm.pkl'), 'rb') as file:
        diff_fundicrest_dpfs = pickle.load(file)
    list_diff_fundicrest_dpfs.append(diff_fundicrest_dpfs)

# row : alpha, col : sub
list_mean_dev_dpfs = np.transpose(np.array(list_mean_dev_dpfs))
list_var_crest_dpfs = np.transpose(np.array(list_var_crest_dpfs))
list_diff_fundicrest_dpfs = np.transpose(np.array(list_diff_fundicrest_dpfs))


### import metrics DPF003
list_mean_dev_dpf003 = list()
list_var_crest_dpf003 = list()
list_diff_fundicrest_dpf003 = list()

for idx, sub in enumerate(list_subs):
    print(sub)
    ses = list_ses[idx]
    # DEV
    with open(os.path.join(folder_metrics, sub + '_' + ses + '_angle_dev_mean_dpf003.pkl'), 'rb') as file:
        mean_dev_dpf003 = pickle.load(file)
    list_mean_dev_dpf003.append(mean_dev_dpf003)
    # VAR
    with open(os.path.join(folder_metrics, sub + '_' + ses + '_var_crest_dpf003.pkl'), 'rb') as file:
        var_crest_dpf003 = pickle.load(file)
    var_crest_dpf003 = [np.sqrt(vv) for vv in var_crest_dpf003]
    list_var_crest_dpf003.append(var_crest_dpf003)
    # DIFF
    with open(os.path.join(folder_metrics, sub + '_' + ses + '_diff_dunficrest_dpf003.pkl'), 'rb') as file:
        diff_fundicrest_dpf003 = pickle.load(file)
    list_diff_fundicrest_dpf003.append(diff_fundicrest_dpf003)


list_mean_dev_dpf003 = np.transpose(np.array(list_mean_dev_dpf003))
list_var_crest_dpf003 = np.transpose(np.array(list_var_crest_dpf003))
list_diff_fundicrest_dpf003 = np.transpose(np.array(list_diff_fundicrest_dpf003))
# select only for alpha = 0.003
list_mean_dev_dpf003 = list_mean_dev_dpf003[8]
list_mean_dev_dpf003 = [180 - xx for xx in list_mean_dev_dpf003]
list_var_crest_dpf003 = list_var_crest_dpf003[8]
list_diff_fundicrest_dpf003 = list_diff_fundicrest_dpf003[8]


#import metrics SULC
list_mean_dev_sulc = list()
list_var_sulc = list()
list_diff_sulc = list()
for idx, sub in enumerate(list_subs):
    print(sub)
    ses = list_ses[idx]
    # DEV
    with open(os.path.join(folder_metrics, sub + '_' + ses + '_angle_dev_mean_sulc.pkl'), 'rb') as file:
        mean_dev_sulc = pickle.load(file)
    list_mean_dev_sulc.append(mean_dev_sulc)
    # VAR
    with open(os.path.join(folder_metrics, sub + '_' + ses + '_var_crest_sulc.pkl'), 'rb') as file:
        var_sulc = pickle.load(file)
    var_sulc = [np.sqrt(vv) for vv in var_sulc]
    list_var_sulc.append(var_sulc)
    # DIFF
    with open(os.path.join(folder_metrics, sub + '_' + ses + '_diff_dunficrest_sulc.pkl'), 'rb') as file:
        diff_sulc = pickle.load(file)
    list_diff_sulc.append(diff_sulc)

list_mean_dev_sulc = np.array(list_mean_dev_sulc).flatten()
list_var_sulc = np.array(list_var_sulc).flatten()
list_diff_sulc = np.array(list_diff_sulc).flatten()


# We create a dataframe for each metric  sub | depth | value

# df dev dpf star
df = pd.DataFrame(dict(sub = [], depth = [], values = []))


for idx, alpha in enumerate(alphas_dpfstar):
    df_dpfstar_dev = pd.DataFrame(dict(sub = list_subs,
                                       depth = np.repeat(str(alpha), len(list_subs)),
                                       values = list_mean_dev_dpfs[idx]))
    df = pd.concat([df, df_dpfstar_dev])

# df dev sulc
df_sulc_dev = pd.DataFrame(dict(sub = list_subs,
                                depth = np.repeat('sulc', len(list_subs)),
                                values = list_mean_dev_sulc))
df = pd.concat([df, df_sulc_dev])

# df dev dpf standard
df_dpf_dev = pd.DataFrame(dict(sub = list_subs,
                               depth = np.repeat('dpf', len(list_subs)),
                               values = list_mean_dev_dpf003))
df = pd.concat([df, df_dpf_dev])


# df var dpf-star
dfv = pd.DataFrame(dict(sub = [], depth = [], values = []))
for idx, alpha in enumerate(alphas_dpfstar):
    df_dpfstar_var = pd.DataFrame(dict(sub = list_subs,
                                       depth = np.repeat( str(alpha), len(list_subs)),
                                       values = list_var_crest_dpfs[idx]))
    dfv = pd.concat([dfv, df_dpfstar_var])

# df var sulc
df_sulc_var = pd.DataFrame(dict(sub = list_subs,
                                depth = np.repeat('sulc', len(list_subs)),
                                values = list_var_sulc))
dfv = pd.concat([dfv, df_sulc_var])

# df var dpf standard
df_dpf_var = pd.DataFrame(dict(sub = list_subs,
                               depth = np.repeat('dpf', len(list_subs)),
                               values = list_var_crest_dpf003))
dfv = pd.concat([dfv, df_dpf_var])

# df diff dpf-star
dff =pd.DataFrame(dict(sub = [], depth = [], values = []))
for idx, alpha in enumerate(alphas_dpfstar):
    df_dpfstar_diff = pd.DataFrame(dict(sub = list_subs,
                                        depth = np.repeat(str(alpha), len(list_subs)),
                                        values = list_diff_fundicrest_dpfs[idx]))

    dff = pd.concat([dff, df_dpfstar_diff])

# df diff sulc
df_sulc_diff = pd.DataFrame(dict(sub = list_subs,
                                 depth = np.repeat('sulc', len(list_subs)),
                                 values = list_diff_sulc))

dff = pd.concat([dff, df_sulc_diff])

# df diff dpf standard
df_dpf_diff = pd.DataFrame(dict(sub = list_subs,
                                depth = np.repeat('dpf', len(list_subs)),
                                values = list_diff_fundicrest_dpf003))
dff = pd.concat([dff, df_dpf_diff])


# figure
fig, ax  = plt.subplots(3)
sns.violinplot(data = df, y = 'values', x = "depth", ax=ax[0])
ax[0].grid()
ax[0].set_title('mean angular deviation across 13 subjects (no adult)')


sns.violinplot(data = dfv, y = 'values', x = "depth",ax=ax[1])
ax[1].grid()
ax[1].set_title('variance crest across 13 subjects (no adult)')


sns.violinplot(data = dff, y = 'values', x = "depth",ax=ax[2])
ax[2].grid()
ax[2].set_title('diff median fundi crest across 13 subjects (no adult)')

plt.tight_layout()
plt.show()