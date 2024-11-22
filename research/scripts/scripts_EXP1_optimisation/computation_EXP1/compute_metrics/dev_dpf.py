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
alphas = [0, 0.0001, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.5]


dev_dpf  = list()
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
    grad_surface = pd.read_csv(os.path.join(wd,'data_EXP1/result_EXP1/depth',sub + '_'+ ses, 'dmap_crest/derivative',
                                            sub + '_' + ses + '_grad_dmap_crest.csv'))

    # import gradient dpf
    grad_dpfs = pd.read_csv(os.path.join(wd, 'data_EXP1/result_EXP1/depth', sub + '_' + ses, 'dpf/derivative',
                                            sub + '_' + ses + '_grad_dpf.csv'))
     # init
    list_dpf= list()

    for jdx, alpha in enumerate(alphas):
        grad_dpf = grad_dpfs[grad_dpfs['alpha']==alpha][['x', 'y', 'z']]

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
                    angle_dpfstar = angle_between(-grad_surface.iloc[vtx].values, grad_dpf.iloc[vtx].values)
                    list_trace_dpfstar.append(angle_dpfstar)
                list_subfundi_dpfstar.append(list_trace_dpfstar)
            list_angle_alpha.append(list_subfundi_dpfstar)

        temp = [item for sublist in list_angle_alpha for item in sublist]
        all = [item for sublist in temp for item in sublist]
        mean_alpha = np.mean(all)
        list_dpf.append(mean_alpha)
    dev_dpf.append(list_dpf)

# save df

df_dpf = pd.DataFrame(dict(subject = np.repeat(subjects, len(alphas)),
                               sessions = np.repeat(sessions, len(alphas)),
                               alphas = np.repeat([alphas], len(subjects), axis=0).ravel(),
                               dev_dpf= np.array(dev_dpf).ravel()))
df_dpf.to_csv(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/dpf/dpfs_dev.csv'), index=False)