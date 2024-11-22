import numpy as np
import pandas as pd
import networkx as nx
import math

def get_labels_lines(path_table_labels_lines):
    """
    This function give the label associated to the name of lines labelised on texture mesh
    we need to read in our project:
    fundi : 50
    crest : 100
    wp_cs : 200
    wp_postcs : 250
    wp_sts : 300
    wp_cg : 350
    wp_cm : 400
    wp_op : 450
    :param path_table_labels_lines: path of the csv file with columns [name_line, label_line]
    :return: dataframe corresponding to the input csv file
    """
    df_labels_lines = pd.read_csv(path_table_labels_lines, index_col='name_line')
    return df_labels_lines

def get_wallpinches_info(path_table_wallpinches_info):
    """
    There is differents number of wallpinches labelised per regions and persubject
    (for exemple : sub A has 2 wallpinches in Central sulcus respectivelly labelised 200,210 while sub B
    has 3 wallpinches in Central Sulcus respectivelly labelised 200,210,220)
    The table_wall_pinches_info give the number of wallpinches per region and per subjects in the study.
    :param path_table_wallpinches_info: path of the csv file with the number of labelised wallpinches
    per regions and per subjects
    :return: dataframe corresponding to the input csv file
    """
    df_wallpinches_info = pd.read_csv(path_table_wallpinches_info)
    return df_wallpinches_info


def vertices_from_label(sub, ses, texture, df_labels_lines, df_wp_info):
    """
    :param sub : string corresponding to the identifiant of the subject we want to get the list of vertices per label
    :param ses : string corresponding to the session identifiant of the subject
    :param texture : gii texture file corresponding to the manual labelisation of the lines {crest, fundi, wall pinches}
    on a mesh.
    :param df_labels_lines: dataframe obtained thanks to the function 'get_labels_lines'
    :param df_wp_info : dataframe obtained thanks to the function 'get_wallpinches_info'
    :return: dict_texture : nested dictionary with label and vertices correponding to the name line.
    fundi : label(=50) and vertices (1D list of vertices)
    crest : label(=100) and vertices (1D list of vertices)
    wp_cs : label(start at 200 with increment of 10) and vertices (2D list of vertices)
    wp_postcs : same ( start at 250)
    ...
    """
    texture = np.array([int(np.round(tex)) for tex in texture])
    #print(np.unique(texture))
    dict_texture = df_labels_lines.to_dict('index')
    labels_name = list(dict_texture.keys())
    for idt in labels_name:
        #print(idt)
        # upload mask
        label = dict_texture[idt]['label']
        #print(label)
        if (idt == 'fundi') | (idt == 'crest'):
            #print('true')
            vertices = np.where(texture == label)[0]
            #print(vertices)
        #else:
        #    nb = df_wp_info[(df_wp_info['sub_id'] == sub) & (df_wp_info['ses_id'] == ses)][idt].values[0]
        #    vertices = [np.where(texture == label + i * 10)[0] for i in np.arange(nb)]
            #print(vertices)
        dict_texture[idt]['vertices'] = vertices
    return dict_texture




def vertices_from_label_2(sub, ses, texture, df_labels_lines):
    """
    :param sub : string corresponding to the identifiant of the subject we want to get the list of vertices per label
    :param ses : string corresponding to the session identifiant of the subject
    :param texture : gii texture file corresponding to the manual labelisation of the lines {crest, fundi, wall pinches}
    on a mesh.
    :param df_labels_lines: dataframe obtained thanks to the function 'get_labels_lines'
    :param df_wp_info : dataframe obtained thanks to the function 'get_wallpinches_info'
    :return: dict_texture : nested dictionary with label and vertices correponding to the name line.
    fundi : label(=50) and vertices (1D list of vertices)
    crest : label(=100) and vertices (1D list of vertices)
    wp_cs : label(start at 200 with increment of 10) and vertices (2D list of vertices)
    wp_postcs : same ( start at 250)
    ...
    """
    texture = np.array([int(np.round(tex)) for tex in texture])
    #print(np.unique(texture))
    dict_texture = df_labels_lines.to_dict('index')
    labels_name = list(dict_texture.keys())
    for idt in labels_name:
        #print(idt)
        # upload mask
        label = dict_texture[idt]['label']
        #print(label)
        if (idt == 'fundi') | (idt == 'crest'):
            #print('true')
            vertices = np.where(texture == label)[0]
            #print(vertices)
        dict_texture[idt]['vertices'] = vertices
    return dict_texture


def sort_vertex_connectivity(adj, mesh, idx_vertices):
    """
    :param adj: ajdacent matrix of the mesh
    :param mesh : Trimesh mesh
    :param idx_vertices : list of index of vertices
    :return: sorted_idx_vertices
    """
    print(idx_vertices)
    adj = adj.tocsr()[idx_vertices,:]
    adj = adj[:,idx_vertices]
    coord_vertices = mesh.vertices[idx_vertices]
    center_mass = mesh.center_mass
    argtop = np.argmax([math.dist(center_mass, vtx) for vtx in coord_vertices])
    top = idx_vertices[argtop]
    print(top)
    argbottom = np.argmin([math.dist(center_mass, vtx) for vtx in coord_vertices])
    bottom = idx_vertices[argbottom]
    print(bottom)
    G = nx.from_numpy_matrix(adj)
    vertex_sorted = nx.shortest_path(G, source = argtop, target=argbottom)
    sorted_idx_vertices = idx_vertices[vertex_sorted]
    sorted_idx_vertices = np.array(sorted_idx_vertices)
    return sorted_idx_vertices


