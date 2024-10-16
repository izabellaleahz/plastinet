# plastinet/analysis/attention_analysis.py
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Dict
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, add_self_loops, k_hop_subgraph

from torch_geometric.data import Data
from sklearn.metrics import pairwise_distances
import numpy as np
import torch

def construct_edge_index_from_spatial(adata, distance_threshold=50):

    spatial_coords = adata.obsm['spatial']

    distances = pairwise_distances(spatial_coords)
    
    mask = (distances < distance_threshold) & (distances > 0)
    
    edge_index = np.array(np.nonzero(mask))
    
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    return edge_index

def adata_to_pyg_data_with_edges(adata, edge_index=None, method='distance', k=10, distance_threshold=50):

    x = torch.tensor(adata.X, dtype=torch.float)
    pos = torch.tensor(adata.obsm['spatial'], dtype=torch.float)

    if edge_index is None:
        if method == 'knn':
            edge_index = construct_edge_index_from_knn(adata, k=k)
        elif method == 'distance':
            edge_index = construct_edge_index_from_spatial(adata, distance_threshold=distance_threshold)
    
    data = Data(x=x, pos=pos, edge_index=edge_index)
    
    return data


def return_edges_in_k_hop(
    data: Data, target_idx: int, hop: int, self_loops: bool = False, return_as_tensor: bool = False
) -> List[Tuple[int, int]]:
    '''
    Exstracts subgraph - returns all edges in k hop 
    '''
    if hop <= 0:
        raise ValueError("Hop must be greater than 0")
    edge_index = add_self_loops(remove_self_loops(data.edge_index)[0])[0] if self_loops else remove_self_loops(data.edge_index)[0]

    _, _, _, inv = k_hop_subgraph(node_idx=target_idx, num_hops=hop, edge_index=edge_index, relabel_nodes=True)
    return edge_index[:, inv].t().tolist() if not return_as_tensor else edge_index[:, inv]

@torch.no_grad()
def generate_att_dict(model, data, sparse: bool = False) -> Dict:
    '''
    Constructs attention matrix for access.
    Extracts attention weights from your model.
    '''

    model(data.x, data.edge_index)
    
    att1 = model.self_attention_weights
    att2 = model.neighbor_attention_weights

    if att1 is None or att2 is None:
        raise AttributeError("Attention weights are not populated in the model.")
    
    att_matrix_dict = {}
    device = att1.device 
    att_matrix_dict[0] = torch.zeros((data.num_nodes, data.num_nodes), device=device)
    att_matrix_dict[1] = torch.zeros((data.num_nodes, data.num_nodes), device=device)
    
    for i in range(att1.size(0)): 
        att_matrix_dict[0][data.edge_index[0, i], data.edge_index[1, i]] = att1[i].mean(dim=0)
    for i in range(att2.size(0)):  
        att_matrix_dict[1][data.edge_index[0, i], data.edge_index[1, i]] = att2[i].mean(dim=0)

    return att_matrix_dict

def prep_for_gatt(model, data, num_hops: int = 2, sparse: bool = False) -> Tuple[Dict, Dict]:
    '''
    Computes the attention matrices as well as correction matrices to account for multi-hop message passing in the graph.
    '''
    att_matrix_dict = generate_att_dict(model=model, data=data, sparse=sparse)
    
    if num_hops > len(att_matrix_dict):
        raise ValueError(f"num_hops ({num_hops}) exceeds the number of attention layers ({len(att_matrix_dict)}).")
    
    correction_matrix_dict = {0: torch.eye(data.num_nodes).to(data.x.device)}  
    for idx in range(1, num_hops):
        if correction_matrix_dict[idx - 1].shape[1] != att_matrix_dict[num_hops - idx].shape[0]:
            raise ValueError(f"Shape mismatch: Correction matrix shape {correction_matrix_dict[idx - 1].shape} "
                             f"and attention matrix shape {att_matrix_dict[num_hops - idx].shape}")
        correction_matrix_dict[idx] = correction_matrix_dict[idx - 1] @ att_matrix_dict[num_hops - idx]

    return att_matrix_dict, correction_matrix_dict


def avgatt(target_edge: Tuple[int, int], att_matrix_dict: Dict, num_hops: int = 2) -> float:
    '''
    Calculates the average attention weight for a specific edge across all layers.
    '''
    return sum(att_matrix_dict[m][target_edge[1], target_edge[0]].item() for m in range(num_hops)) / num_hops

def get_avgatt(target_node, model, data, sparse: bool = True) -> Tuple[List[float], List[Tuple[int, int]]]:
    '''
    Returns the average attention scores instead of the more complex GATT scores.
    '''
    num_hops = model.num_layers  # 2 layers
    att_matrix_dict, _ = prep_for_gatt(model=model, data=data, num_hops=num_hops, sparse=sparse)

    edges_in_k_hop = return_edges_in_k_hop(data=data, target_idx=target_node, hop=num_hops, self_loops=True)
    avgatt_list = [avgatt(current_edge, att_matrix_dict, num_hops) for current_edge in edges_in_k_hop]

    return avgatt_list, edges_in_k_hop


def gatt(target_edge: Tuple[int, int], ref_node: int, att_matrix_dict: Dict, correction_matrix_dict: Dict, num_hops: int = 2) -> float:
    '''
    Calculate the GATT score by considering multi-hop attention information.
    '''
    src_idx, tgt_idx = target_edge
 
    gatt_score = (
        correction_matrix_dict[num_hops - 2][ref_node, tgt_idx].item() * att_matrix_dict[0][tgt_idx, src_idx].item()
        + att_matrix_dict[1][tgt_idx, src_idx].item()
    )
    return gatt_score

def get_gatt(target_node, model, data, sparse: bool = True) -> Tuple[List[float], List[Tuple[int, int]]]:
    '''
    Returns the GATT values for a target node, taking into account all edges in the k-hop neighborhood.
    '''
    num_hops = model.num_layers  # Your model has 2 layers
    att_matrix_dict, correction_matrix_dict = prep_for_gatt(model=model, data=data, num_hops=num_hops, sparse=sparse)

    edges_in_k_hop = return_edges_in_k_hop(data=data, target_idx=target_node, hop=num_hops, self_loops=True)
    gatt_list = [gatt(current_edge, target_node, att_matrix_dict, correction_matrix_dict, num_hops) for current_edge in edges_in_k_hop]

    return gatt_list, edges_in_k_hop
