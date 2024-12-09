# plastinet/data/data_loader.py
import torch
from torch_geometric.data import Data

from .graph_utils import graph_alpha, graph_to_sparse_matrix, sparse_mx_to_torch_edge_list

def create_data_objects(adata, sample_key, radius):
    data_list = []
    spatial_coords = adata.obsm['spatial']

    for s in set(adata.obs[sample_key]):
        print(s)
        sample = adata[adata.obs[sample_key] == s]
        
        spatial_coords = sample.obsm['spatial']
        spatial_graph = graph_alpha(spatial_coords, radius)
        
        edge_index = sparse_mx_to_torch_edge_list(graph_to_sparse_matrix(spatial_graph)).long()
        x = torch.tensor(sample.X.toarray(), dtype=torch.float) if hasattr(sample.X, 'toarray') else torch.tensor(sample.X, dtype=torch.float)
        pos = torch.tensor(spatial_coords, dtype=torch.float)
        
        # Convert cell_ids to a list of strings
        cell_ids = sample.obs.index.to_list()
        
        data = Data(x=x, edge_index=edge_index, pos=pos)
        data.cell_id = cell_ids  # Store as list of strings

        data_list.append(data)
    return data_list
