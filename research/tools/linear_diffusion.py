"""
.. _example_constrained_diffusion:

===================================
Distance maps to a set of points with gdist and networkx (faster)
===================================
"""

# Authors: Julien Lefevre <julien.lefevre@univ-amu.fr>

# License: BSD (3-clause)
# sphinx_gallery_thumbnail_number = 2

###############################################################################
import slam.io as sio
import time
import slam.texture as st
import numpy as np
import gdist
import networkx as nx


def dijkstra_length(mesh, start_indices):
    lengths, ga = dijkstra_lengths(mesh, start_indices)
    length = np.min(lengths,axis=1)
    return length


def dijkstra_lengths(mesh,start_indices):
    # Intermediate step to compute all distances from start_indices to the other indices
    # Return as much distance maps as start_indices
    # edges without duplication

    mod = 1
    if len(start_indices) > 100:
        mod = 10

    edges = mesh.edges_unique

    # the actual length of each unique edge
    length = mesh.edges_unique_length
    print(length)

    # create the graph with edge attributes for length
    # g = nx.Graph()
    # for edge, L in zip(edges, length):
    #     g.add_edge(*edge, length=L)

    # alternative method for weighted graph creation
    # you can also create the graph with from_edgelist and
    # a list comprehension, which is like 1.5x faster
    ga = nx.from_edgelist([(e[0], e[1], {"length": L}) for e, L in zip(edges, length)])
    length_dijk = np.zeros((len(mesh.vertices),len(start_indices)))
    for i, vert_id in enumerate(start_indices):
        dict_length = nx.single_source_dijkstra_path_length(ga, vert_id, weight = "length")
        for key in dict_length.keys():
            length_dijk[key, i] = dict_length[key]
        if i % mod == 0:
            print(i)
    return length_dijk, ga


def gdist_length(mesh, start_indices):
    """
    This func computes the geodesic distance from several points to all vertices on
     mesh by using the gdist.compute_gdist().
    Actually, you can get the geo-distances that you want by changing the
    source and target vertices set.
    :param mesh: trimesh object
    :param vert_id: the point indices
    :return:
    """
    vert = mesh.vertices
    poly = mesh.faces.astype(np.int32)
    length_gdist = np.zeros((len(vert,),len(start_indices)))

    target_index = np.linspace(0, len(vert) - 1, len(vert)).astype(np.int32)

    for i, vert_id in enumerate(start_indices):
        source_index = np.array([vert_id], dtype=np.int32)
        length_gdist[:, i] = gdist.compute_gdist(vert, poly, source_index, target_index)

    return length_gdist

