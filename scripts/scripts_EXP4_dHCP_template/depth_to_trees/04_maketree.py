import slam.io as sio
import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.algorithms.dag import transitive_closure
import slam.texture as stex
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
#list_colors = list(mcolors.TABLEAU_COLORS.keys())
list_colors = list(mcolors.CSS4_COLORS.keys())


# paths
wd = '/home/maxime/callisto/repo/paper_sulcal_depth'
folder_dHCP = '/media/maxime/Expansion/rel3_dHCP'

# init sub
#sub = 'CC00672BN13'
#ses = '200000'

sub = 'CC00576XX16'
ses = '163200'

# load mesh
mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-left_wm.surf.gii'
mesh_folder = os.path.join('/media/maxime/Expansion/rel3_dHCP/', 'sub-' + sub, 'ses-' + ses, 'anat')
mesh_path = os.path.join(mesh_folder, mesh_name)
mesh = sio.load_mesh(mesh_path)

K1 = sio.load_texture(os.path.join(wd, 'data_EXP3/result_EXP3/curvature', sub + '_' + ses + '_K1.gii')).darray[0]
K2 = sio.load_texture(os.path.join(wd, 'data_EXP3/result_EXP3/curvature', sub + '_' + ses + '_K2.gii')).darray[0]
curv = 0.5 * (K1 + K2)

# import levels
save_folder = os.path.join(wd, 'data_EXP4/result_EXP4',sub + '_' + ses )
levels = sio.load_texture(os.path.join( os.path.join(save_folder, sub + '_' + ses + '_sulci_extraction.gii')))
levels = levels.darray[0:29]
last_rank_node  = np.unique(levels[-1])
last_rank_node = last_rank_node[last_rank_node !=0].tolist()
# construct graph
G=nx.DiGraph()

for i in np.arange(len(levels)-1):
    print(i)
    # get level 1 and level2, two successiv levels
    lvl1 = levels[i]
    lvl2 = levels[i+1]

    # gets list of labels for level 1 and level 2
    labels1 = np.unique(lvl1)
    labels1 = labels1[labels1 !=0].tolist()
    labels2 = np.unique(lvl2)
    labels2 = labels2[labels2 !=0].tolist()

    # get all pair of labels between the two levels
    pairs = list(itertools.product(labels1, labels2))
    inclusion = list()
    for idx, pair in enumerate(pairs) :
        inclusion.append( set(np.where(lvl1==pair[0])[0].tolist()).issubset(set(np.where(lvl2==pair[1])[0].tolist())) )

    # for visu
    #df = pd.DataFrame(dict(pairs =  pairs, inclusion = inclusion))
    # for define edges
    pairs_inclu = np.array(pairs)[inclusion]
    pairs_inclusion = [(p[0],p[1]) for p in pairs_inclu]

    # graph construction
    nodes  = np.hstack([labels1, labels2])
    for idx, nod in enumerate(nodes) :
        G.add_node(nod)

    G.add_edges_from(pairs_inclusion)

## hierarchical display
A = nx.nx_agraph.to_agraph(G)
for idx, lvl in enumerate(levels):
    labels = np.unique(lvl)
    labels = labels[labels != 0]
    print(labels)
    A.add_subgraph(labels, rank='same')


#plt.show()

A.draw(os.path.join(save_folder,'example.png'), prog='dot')
pos = graphviz_layout(G, prog="dot")


def dependency_graph(G, nodes):
    return transitive_closure(G).subgraph(nodes)

intersection = list()
inter2 = list()
for nod in G.nodes:
    parents = list(G.predecessors(nod))
    if len(parents) >1:
        intersection.append(True)
        inter2.append(parents)
    else :
        intersection.append(False)

inter2 = np.unique([item for sublist in inter2 for item in sublist])

color_map = list()
for nod in G.nodes:
    if len(np.intersect1d([nod], inter2 ))>0:
        color_map.append('red')
    else:
        color_map.append('blue')

fig1 = plt.subplots()
#nx.draw(G, with_labels=True, pos=pos,node_color=color_map_sulci)
nx.draw(G, with_labels=True, pos=pos)

G2=nx.DiGraph()
G2_nodes = np.array(G.nodes)[intersection]

G2_nodes = inter2
G2_nodes = np.hstack([G2_nodes, last_rank_node])
for noo in G2_nodes:
    G2.add_node(noo)

G2_edge = dependency_graph(G, G2_nodes)
G2_edge = list(G2_edge.edges)

G2_edge_1 = np.array([G2e[0] for G2e in G2_edge])
G2_edge_2 = np.array([G2e[1] for G2e in G2_edge])


G2_edge_1_clean = np.unique(G2_edge_1)
G2_edge_2_clean = list()
for idx, ed1 in enumerate(G2_edge_1_clean):
    ed2 = np.min(G2_edge_2[G2_edge_1 == ed1])
    G2_edge_2_clean.append(ed2)

G2_edge_2_clean = np.array(G2_edge_2_clean)

G2_edges_clean = [(G2_edge_1_clean[i], G2_edge_2_clean[i]) for i in np.arange(len(G2_edge_1_clean))]

G2.add_edges_from(G2_edges_clean)


A2 = nx.nx_agraph.to_agraph(G2)
for idx, lvl in enumerate(levels):
    labels = np.unique(lvl)
    labels = labels[labels != 0]
    nodeA2 = np.intersect1d(labels, G2_nodes)
    print(nodeA2)
    if len(nodeA2)>0:
        A2.add_subgraph(nodeA2, rank='same')

#A2.draw(os.path.join(save_folder, 'example2.png'), prog='dot')

pos2 = graphviz_layout(G2, prog="dot")

fig2 = plt.subplots()
#nx.draw(G2, with_labels=True, pos = pos2)
nx.draw(G2, with_labels=True, pos = pos)


segmentation = np.zeros(len(mesh.vertices))

list_rank = list()
list_labels = list()
for idx,lvls in enumerate(levels) :
    labels = np.unique(lvls)
    labels = labels[labels != 0]
    list_labels.append(labels)
    list_rank.append(np.repeat(idx,len(labels)))


list_rank = np.array([item for sublist in list_rank for item in sublist])
list_labels = np.array([item for sublist in list_labels for item in sublist])


#for idx, lab in enumerate(list_labels):
#    rank = list_rank[idx]
#    segmentation[np.where(levels[rank]==lab)[0]] = lab

print("yo")
for nod in np.sort(G2_nodes)[::-1]:
    print(nod)
    rank = list_rank[np.where(list_labels==nod)[0][0]]
    segmentation[np.where(levels[rank] == nod)[0]] = nod

sio.write_texture(stex.TextureND(darray=segmentation), os.path.join(save_folder,
                                                                    sub + '_' + ses + '_segmentation.gii'))


def activate_node(g, start_node):
    stack = [start_node]
    ws = []

    while stack:
        node = stack.pop()
        preds = g.predecessors(node)
        stack += preds
        print('%s -> %s' % (node, preds))
        ws.append(node)

    return ws



#A2.node_attr['style'] = 'filled'
for idx, nod in enumerate(last_rank_node):
    print(list_colors[idx])
    preds = activate_node(G2, last_rank_node[idx])
    for pred in preds:
        n = A2.get_node(pred)
        n.attr['style'] = 'filled'
        n.attr['fillcolor'] = list_colors[idx]
        #n.attr['fillcolor'] = 'lightblue2'
        #n.attr = {'color': 'lightblue2', 'style': 'filled'}

A2.draw(os.path.join(save_folder, 'example2.png'), prog='dot')



segmentation_2 = np.zeros(len(mesh.vertices))
for idx, nod in enumerate(last_rank_node):
    print(nod)
    preds = activate_node(G2, last_rank_node[idx])
    for pred in preds :
        rank = list_rank[np.where(list_labels==pred)[0][0]]
        segmentation_2[np.where(levels[rank] == pred)[0]] = idx
sio.write_texture(stex.TextureND(darray=segmentation_2), os.path.join(save_folder,
                                                                    sub + '_' + ses + '_segmentation_2.gii'))




nb_lr = len(last_rank_node)
Z = [np.arange(nb_lr)]
x = np.arange(-0.5, nb_lr , 1)  # len = 11
y = np.arange(-0.5, 1, 1)  # len = 1

cmap = ListedColormap(list_colors[:nb_lr])
fig, ax = plt.subplots(figsize=(505/96,17/96))
ax.pcolormesh(x, y, Z, cmap = cmap)
plt.axis('off')
plt.savefig('/home/maxime/ganymede/brainvisa_install/brainvisa/home/.anatomist/rgb/bassins4.jpg',
            bbox_inches='tight', pad_inches=0)