from networkx.algorithms.dag import transitive_closure



def dependency_graph(G, nodes):
    return transitive_closure(G).subgraph(nodes)



print(dependency_graph(G, [1, 2, 4, 6]).edges)
