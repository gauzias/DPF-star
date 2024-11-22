import settings.path_manager as pm
import slam.io as sio
import pandas as pd
import os
import tools.io as tio
import networkx as nx
import numpy as np
import pandas as pd
import slam.texture as stex
import slam.differential_geometry as sdg
import slam.geodesics as slg
import pickle
import tools.gradient as tgrad

def intersection_2d(A,B):
    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
       'formats':ncols * [A.dtype]}
    C = np.intersect1d(A.view(dtype), B.view(dtype))
    # This last bit is optional if you're okay with "C" being a structured array...
    C = C.view(A.dtype).reshape(-1, ncols)
    return C

# load subject
wd = '/home/maxime/callisto/repo/paper_sulcal_depth/data/diffuse_fundicrest'
folder_info_dataset = pm.folder_info_dataset
folder_meshes = pm.folder_meshes
folder_subject_analysis = pm.folder_subject_analysis
folder_manual_labelisation = pm.folder_manual_labelisation
info_database = pd.read_csv(os.path.join(folder_info_dataset, 'info_database.csv'))
size_ratio = pd.read_csv(os.path.join(folder_info_dataset, 'size_ratio_KKI113.csv'))
list_subs = info_database['participant_id'].values
list_ses = info_database['session_id'].values
path_table_labels_line = os.path.join(pm.folder_info_dataset, 'labels_lines.csv')
df_labels_lines = tio.get_labels_lines(path_table_labels_line)
path_table_wallpinches_info = os.path.join(pm.folder_info_dataset, 'wallpinches_info.csv')
df_wp_info = tio.get_wallpinches_info(path_table_wallpinches_info)


#list_subs = list_subs[6:]
#list_ses = list_ses[6:]

list_subs = [list_subs[5]]
list_ses = [list_ses[5]]
for idx, sub, in enumerate(list_subs):
    print('#################### ', sub)
    ses = list_ses[idx]

    #import mesh
    mesh_name = 'sub-'+ sub +'_ses-' +ses + '_hemi-L_space-T2w_wm.surf.gii'
    mesh_path = os.path.join(folder_meshes, mesh_name)
    mesh = sio.load_mesh(mesh_path)

    # import crete et fundi label
    rater = 'MD_full'
    path_lines = os.path.join(folder_manual_labelisation, rater, sub + '_' + ses + '_lines.gii')
    lines = sio.load_texture(path_lines).darray[0]
    dict_lines = tio.vertices_from_label(sub, ses, lines, df_labels_lines, df_wp_info)
    fundi_all = dict_lines['fundi']['vertices']
    crest = dict_lines['crest']['vertices']


    ## detection des extremités des fundi
    ## detection des deux coté de diffusion des extrémité
    ### detection du point le plus eloigné : relié les extremité au point le plus eloigné
    # et separer crest gauche et crete droite du fundi
    ## faire par fundi connecté

    ### regrouper les fundi par groupe connecté

    edges = mesh.edges_unique
    length = mesh.edges_unique_length


    edges_fundi = [ed for ed in edges if (ed[0] in fundi_all) & (ed[1] in fundi_all)]
    length_fundi = [lg for (lg, ed) in zip(length, edges) if (ed[0] in fundi_all) & (ed[1] in fundi_all)]

    ga_fundi = nx.from_edgelist([(e[0], e[1], {"length": L}) for e, L in zip(edges_fundi, length_fundi)])
    subga_fundi = [ga_fundi.subgraph(c) for c in nx.connected_components(ga_fundi)]

    ga = nx.from_edgelist([(e[0], e[1], {"length": L}) for e, L in zip(edges, length)])

    #diffuse crest
    dif_crest_dijk = np.zeros((len(mesh.vertices),len(crest)))

    if len(crest) > 100:
        mod = 10
    for j, vert_jd in enumerate(crest):
        dict_crest_length = nx.single_source_dijkstra_path_length(ga, vert_jd, weight = "length")
        for key in dict_crest_length.keys():
            dif_crest_dijk[key, j] = dict_crest_length[key]
        if j % mod == 0:
            print(j)

    # gradient of  dmap_crest
    dmap_crest = np.min(dif_crest_dijk, axis=1)
    sio.write_texture(stex.TextureND(darray=dmap_crest), os.path.join(wd, sub + '_' + ses + '_dmap_crest.gii'))

    grad_dmap_crest = tgrad.gradient_texture(dmap_crest, mesh)
    df_grad_dmap_crest = pd.DataFrame(grad_dmap_crest, columns=['x', 'y', 'z'])
    df_grad_dmap_crest.to_csv(os.path.join(wd, sub + '_' + ses + '_grad_dmap_crest.csv'), index=False)

    FC_list = list()
    path_FC_list = list()

    for subf in subga_fundi:
        fundi = np.array(subf.nodes)
        mod = 1
        dif_fundi_dijk = np.zeros((len(mesh.vertices),len(fundi)))
        #diffuse fundi connected
        if len(fundi) > 100:
            mod = 10
        for i, vert_id in enumerate(fundi):
            dict_fundi_length = nx.single_source_dijkstra_path_length(ga, vert_id, weight = "length")
            for key in dict_fundi_length.keys():
                dif_fundi_dijk[key, i] = dict_fundi_length[key]
            if i % mod == 0:
                print(i)


        arg_crest_fundi = np.argmin(dif_crest_dijk[fundi, :], axis=1)
        idx_crest_fundi = np.array([crest[arg] for arg in arg_crest_fundi])

        arg_fundi_crest = np.argmin(dif_fundi_dijk[crest, :], axis=1)
        idx_fundi_crest = np.array([fundi[arg] for arg in arg_fundi_crest])

        # binome : fundi | crest
        binome_crest_fundi = np.vstack([fundi, idx_crest_fundi])
        binome_fundi_crest = np.vstack([idx_fundi_crest, crest])

        bin_fundi_crest = np.array([[binome_fundi_crest[0][i], binome_fundi_crest[1][i]] for i in np.arange(binome_fundi_crest.shape[1])])
        bin_crest_fundi= np.array([[binome_crest_fundi[0][i], binome_crest_fundi[1][i]] for i in np.arange(binome_crest_fundi.shape[1])])

        # binome qui verifie l'allé-retour fundi crest
        FC = intersection_2d(bin_fundi_crest , bin_crest_fundi)

        path_FC = []
        for idx, binome in enumerate(FC):
            print(idx, '/', len(FC))
            pFC = slg.shortest_path(mesh, binome[0], binome[1])
            path_FC.append(pFC)

        FC_list.append(FC)
        path_FC_list.append(path_FC)

    #with open(os.path.join(wd, sub + '_' + ses + '_sulcalwall.pkl'), 'wb') as file:
        # with open(os.path.join(wd, sub + '_' + ses + '_sulcal_call.pkl'), 'rb') as file:
        #    # Call load method to deserialze
        #    myvar = pickle.load(file)
        #
        #    print(myvar)
        # A new file will be created
        pickle.dump(path_FC_list, file)


    #text = np.zeros(len(mesh.vertices))

    #for ss in np.arange(len(subga_fundi)):
    #    path_FC = path_FC_list[ss]
    #    for idx, pp in enumerate(path_FC):
    #        text[pp] = 10 + idx*10

    #text[fundi_all] = 100
    #text[crest] = 500
    #sio.write_texture(stex.TextureND(darray= text), os.path.join(wd, sub + '_' + ses + '_sulcalwall_path.gii'))

