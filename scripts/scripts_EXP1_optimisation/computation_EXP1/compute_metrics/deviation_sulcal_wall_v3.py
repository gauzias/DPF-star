import slam.geodesics as sdg
import pickle
import os
import settings.path_manager as pm
import slam.io as sio
import pandas as pd
import os
import tools.io as tio
import networkx as nx
import numpy as np
import pandas as pd
import slam.texture as stex
import slam.geodesics as slg
import pickle
import math
import matplotlib.pyplot as plt
import seaborn as sns

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in degree between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return math.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def grad_list(mesh, trace):
    grad_swp = list()
    for i in np.arange(1,len(trace)):
        gsw = [trace[i] , mesh.vertices[trace[i]] - mesh.vertices[trace[i-1]]]
        grad_swp.append(gsw)
    return  grad_swp

# wd
wd = '/home/maxime/callisto/repo/paper_sulcal_depth/'
# load datasetEXP1
datasetEXP1 = pd.read_csv(os.path.join(wd, 'datasets/dataset_EXP1.csv'))
# load subjects
subjects = datasetEXP1['participant_id'].values
nbs = len(subjects)
sessions = datasetEXP1['session_id'].values
dataset = datasetEXP1['dataset'].values

# param
alphas = [0, 1, 5, 10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 2000]
alphas = [5000, 10000]
#init
dev_curv  = list()
dev_sulc  = list()
dev_dpf003  = list()


for idx, sub, in enumerate(subjects):
    print(sub)
    ses = sessions[idx]
    dset = dataset[idx]

    #load mesh
    mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-L_space-T2w_wm.surf.gii'
    mesh_path = os.path.join(wd , 'data_EXP1/meshes', mesh_name)
    mesh = sio.load_mesh(mesh_path)

    # import sulcal wall line
    with open(os.path.join(wd, 'data_EXP1/sulcalWall_lines', sub + '_' + ses + '_sulcalwall.pkl'), 'rb') as file:
        sw_lines = pickle.load(file)

    # import gradient of the Surface
    grad_surface = pd.read_csv(os.path.join(wd,'data_EXP1/result_EXP1/depth',sub + '_'+ ses, 'dmap_crest/derivative', sub + '_' + ses + '_grad_dmap_crest.csv'))

    # import gradient curv
    grad_curv = pd.read_csv(os.path.join(wd,'data_EXP1/result_EXP1/depth',sub + '_'+ ses, 'curvature/derivative', sub + '_' + ses + '_grad_curv.csv'))
    # import gradient sulc
    grad_sulc = pd.read_csv(os.path.join(wd, 'data_EXP1/result_EXP1/depth', sub + '_' + ses, 'sulc/derivative',
                                            sub + '_' + ses + '_grad_sulc.csv'))
    # import gradient dpf003
    grad_dpf003 = pd.read_csv(os.path.join(wd, 'data_EXP1/result_EXP1/depth', sub + '_' + ses, 'dpf003/derivative',
                                            sub + '_' + ses + '_grad_dpf003.csv'))

    # init
    list_curv = list()
    list_sulc = list()
    list_dpf003 = list()

    # loop
    for sub_fundi in sw_lines: #sw_lines = [[fundi1], [fundi2],...] ; fundi1 = [[sw1], [sw2],...]
        list_subfundi_curv = list()
        list_subfundi_sulc = list()
        list_subfundi_dpf = list()
        for trace in sub_fundi:
            list_trace_curv = list()
            list_trace_sulc = list()
            list_trace_dpf = list()
            if len(trace) > 5:
                trace = trace[2:-2]
            for vtx in trace:
                # print(-gradSurface.iloc[vtx].values, gradDepth.iloc[vtx].values)
                angle_curv = angle_between(-grad_surface.iloc[vtx].values, grad_curv.iloc[vtx].values)
                angle_sulc = angle_between(-grad_surface.iloc[vtx].values, grad_sulc.iloc[vtx].values)
                angle_dpf = angle_between(-grad_surface.iloc[vtx].values, grad_dpf003.iloc[vtx].values)

                list_trace_curv.append(angle_curv)
                list_trace_sulc.append(angle_sulc)
                list_trace_dpf.append(angle_dpf)
            list_subfundi_curv.append(list_trace_curv)
            list_subfundi_sulc.append(list_trace_sulc)
            list_subfundi_dpf.append(list_trace_dpf)
        list_curv.append(list_subfundi_curv)
        list_sulc.append(list_subfundi_sulc)
        list_dpf003.append(list_subfundi_dpf)

    # compute mean
    temp_curv = [item for sublist in list_curv for item in sublist]
    all_angle_curv = [item for sublist in temp_curv for item in sublist]
    mean_curv = np.mean(all_angle_curv)
    dev_curv.append(mean_curv)

    temp_sulc = [item for sublist in list_sulc for item in sublist]
    all_angle_sulc = [item for sublist in temp_sulc for item in sublist]
    mean_sulc = np.mean(all_angle_sulc)
    dev_sulc.append(mean_sulc)

    temp_dpf = [item for sublist in list_dpf003 for item in sublist]
    all_angle_dpf = [item for sublist in temp_dpf for item in sublist]
    mean_dpf = np.mean(all_angle_dpf)
    dev_dpf003.append(mean_dpf)

# save df
df_curv = pd.DataFrame(dict(subject = subjects,
                            sessions = sessions,
                            angle_curv = dev_curv))
df_curv.to_csv(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/curv/curv_dev.csv' ), index=False)

df_sulc = pd.DataFrame(dict(subject = subjects,
                            sessions = sessions,
                            angle_sulc = dev_sulc))
df_sulc.to_csv(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/sulc/sulc_dev.csv' ), index=False)

df_dpf003 = pd.DataFrame(dict(subject = subjects,
                            sessions = sessions,
                            angle_dpf003 = dev_dpf003))
df_dpf003.to_csv(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/dpf003/dpf003_dev.csv' ), index=False)


## dpfstar
dev_dpfstar  = list()
for idx, sub, in enumerate(subjects):
    print(sub)
    ses = sessions[idx]
    dset = dataset[idx]

    #load mesh
    mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-L_space-T2w_wm.surf.gii'
    mesh_path = os.path.join(wd , 'data_EXP1/meshes', mesh_name)
    mesh = sio.load_mesh(mesh_path)

    # import sulcal wall line
    with open(os.path.join(wd, 'data_EXP1/sulcalWall_lines', sub + '_' + ses + '_sulcalwall.pkl'), 'rb') as file:
        sw_lines = pickle.load(file)

    # import gradient of the Surface
    grad_surface = pd.read_csv(os.path.join(wd,'data_EXP1/result_EXP1/depth',sub + '_'+ ses, 'dmap_crest/derivative', sub + '_' + ses + '_grad_dmap_crest.csv'))

    # import gradient dpfstar
    grad_dpfstars = pd.read_csv(os.path.join(wd, 'data_EXP1/result_EXP1/depth', sub + '_' + ses, 'dpfstar_surface/derivative',
                                            sub + '_' + ses + '_grad_dpfstar_2.csv'))
     # init
    list_dpfstar= list()

    for jdx, alpha in enumerate(alphas):
        grad_dpfstar = grad_dpfstars[grad_dpfstars['alpha']==alpha][['x', 'y', 'z']]

        # init
        list_angle_alpha = list()

        # loop
        for sub_fundi in sw_lines:  # sw_lines = [[fundi1], [fundi2],...] ; fundi1 = [[sw1], [sw2],...]
            list_subfundi_dpfstar = list()
            for trace in sub_fundi:
                list_trace_dpfstar = list()
                if len(trace) > 5:
                    trace = trace[2:-2]
                for vtx in trace:
                    # print(-gradSurface.iloc[vtx].values, gradDepth.iloc[vtx].values)
                    angle_dpfstar = angle_between(-grad_surface.iloc[vtx].values, grad_dpfstar.iloc[vtx].values)
                    list_trace_dpfstar.append(angle_dpfstar)
                list_subfundi_dpfstar.append(list_trace_dpfstar)
            list_angle_alpha.append(list_subfundi_dpfstar)

        temp = [item for sublist in list_angle_alpha for item in sublist]
        all = [item for sublist in temp for item in sublist]
        mean_alpha = np.mean(all)
        list_dpfstar.append(mean_alpha)
    dev_dpfstar.append(list_dpfstar)

# save df

df_dpfstar = pd.DataFrame(dict(subject = np.repeat(subjects, len(alphas)),
                               sessions = np.repeat(sessions, len(alphas)),
                               alphas = np.repeat([alphas], len(subjects), axis=0).ravel(),
                               angle_dpfstar= np.array(dev_dpfstar).ravel()))
df_dpfstar.to_csv(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/dpfstar_surface/dpfstar_dev_2.csv'), index=False)







