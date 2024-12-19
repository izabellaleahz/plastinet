# plastinet/data/graph_utils.py
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
import torch

def graph_alpha(spatial_locs, radius):
    G = nx.Graph()
    kd_tree = cKDTree(spatial_locs)
    
    for i in range(len(spatial_locs)):
        G.add_node(i)
    
    for i, coord in enumerate(spatial_locs):
        indices = kd_tree.query_ball_point(coord, radius)
        for j in indices:
            if i != j:
                G.add_edge(i, j)
    return G

def graph_to_sparse_matrix(G):
    n_nodes = G.number_of_nodes()
    edges = np.array(G.edges()).T
    data = np.ones(edges.shape[1])
    sparse_matrix = coo_matrix((data, (edges[0], edges[1])), shape=(n_nodes, n_nodes), dtype=np.float32)
    return sparse_matrix

def sparse_mx_to_torch_edge_list(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    edge_list = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return edge_list
