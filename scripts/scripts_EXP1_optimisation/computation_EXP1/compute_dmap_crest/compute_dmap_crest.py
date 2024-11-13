
import slam.io as sio
import os
import tools.gradient as gradient
import tools.io as tio
import networkx as nx
import numpy as np
import pandas as pd
import slam.texture as stex

# path manager
#wd = '/envau/work/meca/users/dieudonne.m/wd_depth_pipeline'
wd = '/home/maxime/callisto/repo/paper_sulcal_depth'


folder_info_database = os.path.join(wd, 'data/info_database' )
folder_meshes = os.path.join(wd, 'data/meshes')
folder_subject_analysis = os.path.join(wd, 'data/subject_analysis')
folder_manual_labelisation = os.path.join(wd, 'data/manual_labelisation')

# load data
info_database = pd.read_csv(os.path.join(folder_info_database, 'info_database.csv'))
path_table_labels_line = os.path.join(folder_info_database, 'labels_lines.csv')
df_labels_lines = tio.get_labels_lines(path_table_labels_line)

subjects = ['KKI2009-142', 'KKI2009-505']
sessions = ['MR1','MR1']


#subjects = info_database['participant_id'].values[0:1]
#sessions = info_database['session_id'].values[0:1]


for idx, sub, in enumerate(subjects):
    print(sub)
    print('.... Computing dmap crest')
    ses = sessions[idx]
    mesh_name = 'sub-'+ sub +'_ses-' +ses + '_hemi-L_space-T2w_wm.surf.gii'
    mesh_path = os.path.join(folder_meshes, mesh_name)
    mesh = sio.load_mesh(mesh_path)

    # import crete label
    rater = 'MD_full'
    path_lines = os.path.join(folder_manual_labelisation, rater, sub + '_' + ses + '_lines.gii')
    lines = sio.load_texture(path_lines).darray[0]
    dict_lines = tio.vertices_from_label_2(sub, ses, lines, df_labels_lines)
    crest = dict_lines['crest']['vertices']

    # converting mesh into graph network
    edges = mesh.edges_unique
    length = mesh.edges_unique_length
    ga = nx.from_edgelist([(e[0], e[1], {"length": L}) for e, L in zip(edges, length)])

    # compute dmap crest with dijkstra
    dif_crest_dijk = np.zeros((len(mesh.vertices),len(crest)))
    if len(crest) > 100:
        mod = 10
    for j, vert_jd in enumerate(crest):
        dict_crest_length = nx.single_source_dijkstra_path_length(ga, vert_jd, weight = "length")
        for key in dict_crest_length.keys():
            dif_crest_dijk[key, j] = dict_crest_length[key]
        if j % mod == 0:
            print(j)

    arg_dmap_crest = np.argmin(dif_crest_dijk,axis = 1) # argument of the nearest crest
    dmap_crest = np.min(dif_crest_dijk, axis= 1) # distance of the nearest crest
    # save
    folder_dmap_crest = os.path.join(folder_subject_analysis, sub + '_' + ses, 'depth_metric', 'dmap_crest')
    fname_dmap_crest = sub + '_' + ses + '_dmap_crest_test.gii'
    fname_arg_dmap_crest = sub + '_' + ses + '_arg_dmap_crest_test.gii'
    if not os.path.exists(folder_dmap_crest):
        os.makedirs(folder_dmap_crest)
    sio.write_texture(stex.TextureND(darray=arg_dmap_crest), os.path.join(folder_dmap_crest, fname_arg_dmap_crest))
    sio.write_texture(stex.TextureND(darray=dmap_crest), os.path.join(folder_dmap_crest, fname_dmap_crest))


    # compute Gradient of dmap_crest
    print('.... Computing gradient dmap crest')
    grad_dmap_crest = gradient.gradient_texture(dmap_crest, mesh)
    df_grad_dmap_crest = pd.DataFrame(grad_dmap_crest, columns=['x', 'y', 'z'])
    df_grad_dmap_crest.to_csv(os.path.join(folder_dmap_crest, sub + '_' + ses + '_grad_dmap_crest.csv'), index=False)
