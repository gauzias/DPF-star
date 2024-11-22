import slam.io as sio
import os
import pandas as pd
import numpy as np

"""
date : 10/08/2023
author : maxime.dieudonne@univ-amu.fr

this script compute std crest for curv
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
alphas = [0, 0.0001, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.5]

# init
list_std_crest_dpf = list()

for idx, sub in enumerate(subjects):
    print(sub)
    ses = sessions[idx]
    dset = dataset[idx]

    # import dpfstar
    dpf_path = os.path.join(wd, 'data_EXP1/result_EXP1/depth', sub + '_' + ses, 'dpf', sub + '_' + ses + '_dpf.gii')
    dpfs = sio.load_texture(dpf_path).darray

    # import lines
    path_lines = os.path.join(wd, 'data_EXP1/manual_labelisation/MD_full', sub + '_' + ses + '_lines.gii')
    lines = sio.load_texture(path_lines).darray[0]
    lines = np.array([np.round(li).astype(int) for li in lines])
    crest = np.where(lines == 100)[0]
    fundi = np.where(lines == 50)[0]


    # dpfstar
    # load crest and fundi
    for jdx, alpha in enumerate(alphas):
        dpf = dpfs[jdx]
        dpf_crest = dpf[crest]
        dpf_fundi = dpf[fundi]
        # compute std crest dpfstar
        std_crest_dpf = np.std(dpf_crest) / np.abs((np.percentile(dpf, 95) - np.percentile(dpf, 5)))
        list_std_crest_dpf.append(std_crest_dpf)


df_dpf = pd.DataFrame(dict(subject = np.repeat(subjects, len(alphas)),
                               sessions = np.repeat(sessions, len(alphas)),
                               alphas = np.repeat([alphas], len(subjects), axis=0).ravel(),
                               std_crest_dpf = list_std_crest_dpf))

folder_save = os.path.join(wd, 'data_EXP1/result_EXP1/metrics/dpf')
if not os.path.exists(folder_save):
    os.makedirs(folder_save)
df_dpf.to_csv(os.path.join(folder_save, 'dpf_std_crest.csv' ), index=False)


