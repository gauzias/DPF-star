from pathlib import Path
import seaborn as sns
import numpy as np
from research.tools import rw as rw
import pandas as pd
import research.tools.io as tio
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
import matplotlib.cm
from matplotlib.colors import LinearSegmentedColormap

# Date: 25/07/2023, mise a jour : 06/05/2025
# Description: This script generates the figure of Experiment 1: optimization of alpha for the DPF-star method.

# Params
alphas_dpfstar = [0, 1, 5, 10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 2000]
alphas_dpf003 = [0, 0.0001, 0.002, 0.0003, 0.0004, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.5, 1]

# Paths
wd = Path('/home/maxime/callisto/repo/paper_sulcal_depth')
folder_info_dataset = Path('D:/Callisto/repo/paper_sulcal_depth/datasets')
folder_meshes = Path('D:/Callisto/repo/paper_sulcal_depth/data_EXP1/meshes')
folder_metrics = Path('D:/Callisto/repo/paper_sulcal_depth/data/group_analysis/dev')

# Load subject info
#info_database = pd.read_csv(folder_info_dataset / 'info_database.csv')
info_database = pd.read_csv(folder_info_dataset / 'dataset_EXP1.csv')
list_subs = info_database['participant_id'].values[:-3]
list_ses = info_database['session_id'].values[:-3]
list_scan_sage = info_database['scan_age'].values[:-3]

# Load volume data
list_volume = []
for idx, sub in enumerate(list_subs):
    ses = list_ses[idx]
    mesh_name = f'sub-{sub}_ses-{ses}_hemi-L_space-T2w_wm.surf.gii'
    mesh_path = folder_meshes / mesh_name
    mesh = rw.load_mesh(str(mesh_path))
    list_volume.append(mesh.volume)

# Sort subjects by volume
argsort_volume = np.argsort(list_volume)
list_subs = list_subs[argsort_volume]
list_ses = list_ses[argsort_volume]
list_scan_sage = list_scan_sage[argsort_volume]
list_volume = np.array(list_volume)[argsort_volume]

# Load label info
df_labels_lines = tio.get_labels_lines(folder_info_dataset / 'labels_lines.csv')
df_wp_info = tio.get_wallpinches_info(folder_info_dataset / 'wallpinches_info.csv')

# Helper function to load metrics from pickle
def load_pickle_metric(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Load DPF-star metrics
list_mean_dev_dpfs, list_var_crest_dpfs, list_diff_fundicrest_dpfs = [], [], []

for idx, sub in enumerate(list_subs):
    ses = list_ses[idx]
    list_mean_dev_dpfs.append(load_pickle_metric(folder_metrics / f"{sub}_{ses}_angle_dev_mean.pkl"))
    var = load_pickle_metric(folder_metrics / f"{sub}_{ses}_var_crest_dpf_norm.pkl")
    list_var_crest_dpfs.append([np.sqrt(v) for v in var])
    list_diff_fundicrest_dpfs.append(load_pickle_metric(folder_metrics / f"{sub}_{ses}_diff_dunficrest_dpf_norm.pkl"))

list_mean_dev_dpfs = np.transpose(np.array(list_mean_dev_dpfs))
list_var_crest_dpfs = np.transpose(np.array(list_var_crest_dpfs))
list_diff_fundicrest_dpfs = np.transpose(np.array(list_diff_fundicrest_dpfs))

# Load DPF003 metrics
list_mean_dev_dpf003, list_var_crest_dpf003, list_diff_fundicrest_dpf003 = [], [], []

for idx, sub in enumerate(list_subs):
    ses = list_ses[idx]
    list_mean_dev_dpf003.append(load_pickle_metric(folder_metrics / f"{sub}_{ses}_angle_dev_mean_dpf003.pkl"))
    var = load_pickle_metric(folder_metrics / f"{sub}_{ses}_var_crest_dpf003.pkl")
    list_var_crest_dpf003.append([np.sqrt(v) for v in var])
    list_diff_fundicrest_dpf003.append(load_pickle_metric(folder_metrics / f"{sub}_{ses}_diff_dunficrest_dpf003.pkl"))

list_mean_dev_dpf003 = np.transpose(np.array(list_mean_dev_dpf003))[8]
list_mean_dev_dpf003 = [180 - xx for xx in list_mean_dev_dpf003]
list_var_crest_dpf003 = np.transpose(np.array(list_var_crest_dpf003))[8]
list_diff_fundicrest_dpf003 = np.transpose(np.array(list_diff_fundicrest_dpf003))[8]

# Load sulc metrics
list_mean_dev_sulc, list_var_sulc, list_diff_sulc = [], [], []

for idx, sub in enumerate(list_subs):
    ses = list_ses[idx]
    list_mean_dev_sulc.append(load_pickle_metric(folder_metrics / f"{sub}_{ses}_angle_dev_mean_sulc.pkl"))
    var = load_pickle_metric(folder_metrics / f"{sub}_{ses}_var_crest_sulc.pkl")
    list_var_sulc.append([np.sqrt(v) for v in var])
    list_diff_sulc.append(load_pickle_metric(folder_metrics / f"{sub}_{ses}_diff_dunficrest_sulc.pkl"))

list_mean_dev_sulc = np.array(list_mean_dev_sulc).flatten()
list_var_sulc = np.array(list_var_sulc).flatten()
list_diff_sulc = np.array(list_diff_sulc).flatten()

# Build DataFrames

def build_df(metric_list, alphas, list_subs, method_label):
    df = pd.DataFrame(dict(sub=[], depth=[], values=[]))
    for idx, alpha in enumerate(alphas):
        alpha_df = pd.DataFrame(dict(
            sub=list_subs,
            depth=np.repeat(str(alpha), len(list_subs)),
            values=metric_list[idx]))
        df = pd.concat([df, alpha_df])
    return df

# DEV
df = build_df(list_mean_dev_dpfs, alphas_dpfstar, list_subs, 'dpfstar')
df = pd.concat([df, pd.DataFrame({'sub': list_subs, 'depth': 'sulc', 'values': list_mean_dev_sulc})])
df = pd.concat([df, pd.DataFrame({'sub': list_subs, 'depth': 'dpf', 'values': list_mean_dev_dpf003})])

# VAR
dfv = build_df(list_var_crest_dpfs, alphas_dpfstar, list_subs, 'dpfstar')
dfv = pd.concat([dfv, pd.DataFrame({'sub': list_subs, 'depth': 'sulc', 'values': list_var_sulc})])
dfv = pd.concat([dfv, pd.DataFrame({'sub': list_subs, 'depth': 'dpf', 'values': list_var_crest_dpf003})])

# DIFF
dff = build_df(list_diff_fundicrest_dpfs, alphas_dpfstar, list_subs, 'dpfstar')
dff = pd.concat([dff, pd.DataFrame({'sub': list_subs, 'depth': 'sulc', 'values': list_diff_sulc})])
dff = pd.concat([dff, pd.DataFrame({'sub': list_subs, 'depth': 'dpf', 'values': list_diff_fundicrest_dpf003})])

# Plot
fig, ax = plt.subplots(3)
sns.violinplot(data=df, y='values', x='depth', ax=ax[0])
ax[0].grid()
ax[0].set_title('mean angular deviation across 13 subjects (no adult)')

sns.violinplot(data=dfv, y='values', x='depth', ax=ax[1])
ax[1].grid()
ax[1].set_title('variance crest across 13 subjects (no adult)')

sns.violinplot(data=dff, y='values', x='depth', ax=ax[2])
ax[2].grid()
ax[2].set_title('diff median fundi crest across 13 subjects (no adult)')

plt.tight_layout()
plt.show()
