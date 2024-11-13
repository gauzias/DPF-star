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

sub = list_subs[3]
ses = list_ses[3]

#import mesh
mesh_name = 'sub-'+ sub +'_ses-' +ses + '_hemi-L_space-T2w_wm.surf.gii'
mesh_path = os.path.join(folder_meshes, mesh_name)
mesh = sio.load_mesh(mesh_path)

# import crete et fundi label
rater = 'MD_full'
path_lines = os.path.join(folder_manual_labelisation, rater, sub + '_' + ses + '_lines.gii')
lines = sio.load_texture(path_lines).darray[0]
dict_lines = tio.vertices_from_label(sub, ses, lines, df_labels_lines, df_wp_info)
fundi = dict_lines['fundi']['vertices']
crest = dict_lines['crest']['vertices']


# diffuser le fundi -> tableau
edges = mesh.edges_unique
length = mesh.edges_unique_length
ga = nx.from_edgelist([(e[0], e[1], {"length": L}) for e, L in zip(edges, length)])
mod = 1

dif_fundi_dijk = np.zeros((len(mesh.vertices),len(fundi)))
dif_crest_dijk = np.zeros((len(mesh.vertices),len(crest)))
#diffuse fundi
if len(fundi) > 100:
    mod = 10
for i, vert_id in enumerate(fundi):
    dict_fundi_length = nx.single_source_dijkstra_path_length(ga, vert_id, weight = "length")
    for key in dict_fundi_length.keys():
        dif_fundi_dijk[key, i] = dict_fundi_length[key]
    if i % mod == 0:
        print(i)
#diffuse crest
if len(crest) > 100:
    mod = 10
for j, vert_jd in enumerate(crest):
    dict_crest_length = nx.single_source_dijkstra_path_length(ga, vert_jd, weight = "length")
    for key in dict_crest_length.keys():
        dif_crest_dijk[key, j] = dict_crest_length[key]
    if j % mod == 0:
        print(j)

# dist


dist_crest_fundi = np.min(dif_crest_dijk[fundi, : ], axis=1)
#dist_crest_fundi = dist_crest_fundi[fundi]

dist_fundi_crest = np.min(dif_fundi_dijk[crest, :], axis=1)
#dist_fundi_crest = dist_fundi_crest[crest]

# arg
arg_crest_fundi = np.argmin(dif_crest_dijk[fundi, :], axis=1)
idx_crest_fundi = np.array([crest[arg] for arg in arg_crest_fundi])
#arg_crest_fundi = arg_crest_fundi[fundi]

arg_fundi_crest = np.argmin(dif_fundi_dijk[crest, :], axis=1)
idx_fundi_crest = np.array([fundi[arg] for arg in arg_fundi_crest])

#arg_fundi_crest = arg_fundi_crest[crest]

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


# texture

#for idx, binome in enumerate(FC):
#    text[binome[0]] = 10 + (i*10)
#    text[binome[1]] = 10 + (i*10)
#sio.write_texture(stex.TextureND(darray= text), os.path.join(wd, sub + '_' + ses + '_tadam.gii'))


text = np.zeros(len(mesh.vertices))
for idx, pp in enumerate(path_FC):
    text[pp] = 10 + idx*10

sio.write_texture(stex.TextureND(darray= text), os.path.join(wd, sub + '_' + ses + '_fundicrest_path.gii'))

text[fundi] = 100
text[crest] = 500
sio.write_texture(stex.TextureND(darray= text), os.path.join(wd, sub + '_' + ses + '_fundicrest_path_tadam.gii'))

"""
# appliquer le masque fundi et crest


dmap_crest = dmap_crest[fundi]
dmap_fundi = dmap_fundi[crest]



# keep crest
df_crest_fundi = pd.DataFrame(dict( crest = crest, fundi_plus_proche = arg_dmap_crest,
                                   dist = dmap_crest))
keep_crest = []
keep_associe_fundi = []
fuu = np.unique(df_crest_fundi['fundi_plus_proche'])
for idx, fu in enumerate(fuu) :
    subdf = df_crest_fundi[df_crest_fundi['fundi_plus_proche'] == fu]
    icc = np.argmin(subdf['dist'])
    cc = subdf['crest'][icc]
    kf = subdf['fundi_plus_proche'][icc]

    keep_crest.append(cc)
    keep_associe_fundi.append(kf)


# keep fundi
df_fundi_crest = pd.DataFrame(dict( fundi = fundi, crest_plus_proche = arg_dmap_fundi,
                                   dist = dmap_fundi))
keep_fundi = []
keep_associe_crest = []
crr = np.unique(df_fundi_crest['crest_plus_proche'])
for idx, cr in enumerate(crr) :
    subdff = df_fundi_crest[df_fundi_crest['crest_plus_proche'] == cr]
    iff = np.argmin(subdff['dist'])
    ff = subdff['fundi'][iff]
    kc = subdff['crest_plus_proche'][iff]

    keep_fundi.append(ff)
    keep_associe_crest.append(kc)

ffcc = np.array([[f,c] for (f,c) in zip(keep_fundi, keep_associe_crest)])
ccff = np.array([[f,c] for (f,c) in zip(keep_crest, keep_associe_fundi)])


FC = intersection_2d(ffcc, ccff)

# obtenir une crete unique associé à un fundi unique. C|F
# convertir en graphe le mesh , trouver le plus court chemin geodesique entre C et F.
# gradient sur ces plus court chemin

path_FC = []
for idx, binome in enumerate(FC):
    print(idx, '/', len(FC))
    pFC = slg.shortest_path(mesh, binome[0], binome[1])
    path_FC.append(pFC)


# texture

text = np.zeros(len(mesh.vertices))
for idx, pp in enumerate(path_FC):
    text[pp] = 10 + i*10


sio.write_texture(stex.TextureND(darray= text), os.path.join(wd, sub + '_' + ses + '_fundicrest_path.gii'))


"""


