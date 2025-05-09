from  research.tools import rw as sio
import os
import pandas as pd
import tools.depth as depth
import numpy as np

"""
date : 10/08/2023, mise a jour 06/05/2023
author : maxime.dieudonne@univ-amu.fr

this script compute diff median between fundi and crest
"""

# wd
wd = '/home/maxime/callisto/repo/paper_sulcal_depth/'

# load datasetEXP1
datasetEXP1 = pd.read_csv(os.path.join(wd, 'datasets/dataset_EXP1.csv'))


### COMPUTE CURV

# load subjects
subjects = datasetEXP1['participant_id'].values
sessions = datasetEXP1['session_id'].values
dataset = datasetEXP1['dataset'].values

# param
alphas = [0, 1, 5, 10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 2000]


# init
list_curv = list()
list_sulc = list()
list_dpf003 = list()
list_dpfstar = list()

for idx, sub in enumerate(subjects):
    print(sub)
    ses = sessions[idx]
    dset = dataset[idx]

    # import curv
    K1_path = os.path.join(wd, 'data_EXP1/result_EXP1/depth', sub + '_' + ses,  'curvature',
                           sub + '_' + ses + '_K1.gii')
    K2_path = os.path.join(wd, 'data_EXP1/result_EXP1/depth', sub + '_' + ses,  'curvature',
                           sub + '_' + ses + '_K2.gii')
    K1 = sio.load_texture(K1_path).darray[0]
    K2 = sio.load_texture(K2_path).darray[0]
    curv = 0.5 * (K1 + K2)

    # import sulc
    if dset == 'dHCP':
        sulc_name = 'sub-' + sub + '_ses-'+ ses + '_hemi-L_space-T2w_sulc.shape.gii'
    if dset == 'KKI2009':
        sulc_name = sub + '_' + ses + '_sulc.gii'
    sulc_path = os.path.join(wd, 'data_EXP1/result_EXP1/depth', sub + '_' + ses, 'sulc', sulc_name)
    sulc = sio.load_texture(sulc_path).darray[0]

    # import dpf003
    dpf003_path = os.path.join(wd, 'data_EXP1/result_EXP1/depth', sub + '_' + ses, 'dpf003',
                              sub + '_' + ses + '_dpf003.gii')
    dpf003 = sio.load_texture(dpf003_path).darray[0]

    # import dpfstar
    dpfstar_path = os.path.join(wd, 'data_EXP1/result_EXP1/depth', sub + '_' + ses, 'dpfstar', sub + '_' + ses + '_dpfstar.gii')
    dpfstars = sio.load_texture(dpfstar_path).darray

    # import lines
    path_lines = os.path.join(wd, 'data_EXP1/manual_labelisation/MD_full', sub + '_' + ses + '_lines.gii')
    lines = sio.load_texture(path_lines).darray[0]
    lines = np.array([np.round(li).astype(int) for li in lines])
    crest = np.where(lines == 100)[0]
    fundi = np.where(lines == 50)[0]

    # CURV
    # load crest and fundi
    curv_crest = curv[crest]
    curv_fundi = curv[fundi]
    # compute  diff fundicrest curv
    diff_curv = np.abs(np.median(curv_crest) - np.median(curv_fundi))/\
                np.abs((np.percentile(curv, 95) - np.percentile(curv, 5)))
    list_curv.append(diff_curv)

    # SULC
    # load crest and fundi
    sulc_crest = sulc[crest]
    sulc_fundi = sulc[fundi]
    # compute  diff fundicrest curv
    diff_sulc = np.abs(np.median(sulc_crest) - np.median(sulc_fundi))/\
                np.abs((np.percentile(sulc, 95) - np.percentile(sulc, 5)))
    list_sulc.append(diff_sulc)

    # DPF003
    # load crest and fundi
    dpf003_crest = dpf003[crest]
    dpf003_fundi = dpf003[fundi]
    # compute  diff fundicrest curv
    diff_dpf003 = np.abs(np.median(dpf003_crest) - np.median(dpf003_fundi))/\
                np.abs((np.percentile(dpf003, 95) - np.percentile(dpf003, 5)))
    list_dpf003.append(diff_dpf003)

    # dpfstar
    # load crest and fundi
    for jdx, alpha in enumerate(alphas):
        dpfstar = dpfstars[jdx]
        dpfstar_crest = dpfstar[crest]
        dpfstar_fundi = dpfstar[fundi]
        # compute  diff fundicrest curv
        diff_dpfstar = np.abs(np.median(dpfstar_crest) - np.median(dpfstar_fundi)) / \
                    np.abs((np.percentile(dpfstar, 95) - np.percentile(dpfstar, 5)))
        list_dpfstar.append(diff_dpfstar)


df_curv = pd.DataFrame(dict(subject = subjects, sessions = sessions, diff_fundicrest_curv = list_curv))
df_curv.to_csv(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/curv/curv_diff_fundicrest.csv' ), index=False)

df_sulc = pd.DataFrame(dict(subject = subjects, sessions = sessions, diff_fundicrest_sulc = list_sulc))
df_sulc.to_csv(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/sulc/sulc_diff_fundicrest.csv' ), index=False)

df_dpf003 = pd.DataFrame(dict(subject = subjects, sessions = sessions, diff_fundicrest_dpf003 = list_dpf003))
df_dpf003.to_csv(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/dpf003/dpf003_diff_fundicrest.csv' ), index=False)

df_dpfstar = pd.DataFrame(dict(subject = np.repeat(subjects, len(alphas)),
                               sessions = np.repeat(sessions, len(alphas)),
                               alphas = np.repeat([alphas], len(subjects),axis=0).ravel(),
                               diff_fundicrest_dpfstar = list_dpfstar))
df_dpfstar.to_csv(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/dpfstar/dpfstar_diff_fundicrest.csv' ), index=False)

