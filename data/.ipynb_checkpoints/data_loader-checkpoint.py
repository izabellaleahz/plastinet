# plastinet/data/data_loader.py
import torch
from torch_geometric.data import Data

from .graph_utils import graph_alpha, graph_to_sparse_matrix, sparse_mx_to_torch_edge_list

def create_data_objects(adata, sample_key, radius):
    data_list = []
    for s in set(adata.obs[sample_key]):
        sample = adata[adata.obs[sample_key] == s]
        spatial_coords = sample.obsm['spatial']
        spatial_graph = graph_alpha(spatial_coords, radius)

        edge_index = sparse_mx_to_torch_edge_list(graph_to_sparse_matrix(spatial_graph)).long()
        x = torch.tensor(sample.X.toarray(), dtype=torch.float)
        pos = torch.tensor(spatial_coords, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, pos=pos, cell_ids=sample.obs.index)
        data_list.append(data)
    return data_list