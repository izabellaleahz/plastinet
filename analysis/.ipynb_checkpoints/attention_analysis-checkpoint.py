# plastinet/analysis/attention_analysis.py
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Dict
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, add_self_loops, k_hop_subgraph

def calculate_attention_scores(embedding_adata, neighbor_indices, attn_weights1, attn_weights2, cell_type_obs, cell_type):
    # Attention score calculation logic from your code
    pass

def calculate_attention_correlations(embedding_adata, mean_attention_scores):
    # Correlation calculation logic from your code
    pass


def return_edges_in_k_hop(
    data: Data, target_idx: int, hop: int, self_loops: bool = False, return_as_tensor: bool = False
) -> List[Tuple[int, int]]:
    if hop <= 0:
        raise ValueError("Hop must be greater than 0")
    edge_index = add_self_loops(remove_self_loops(data.edge_index)[0])[0] if self_loops else remove_self_loops(data.edge_index)[0]

    _, _, _, inv = k_hop_subgraph(node_idx=target_idx, num_hops=hop, edge_index=edge_index, relabel_nodes=True)
    return edge_index[:, inv].t().tolist() if not return_as_tensor else edge_index[:, inv]


@torch.no_grad()
def generate_att_dict(model, data, sparse: bool = True) -> Dict:
    model(data.x, data.edge_index, return_att=True)
    att = model.att
    att_matrix_dict = {}
    device = att[0][0].device

    for idx, att_info in enumerate(att):
        if sparse:
            att_matrix_dict[idx] = torch.sparse_coo_tensor(att_info[0], att_info[1].mean(dim=1).squeeze(), size=(data.num_nodes, data.num_nodes), device=device).t()
        else:
            att_matrix_dict[idx] = torch.zeros((data.num_nodes, data.num_nodes)).to(device)
            att_matrix_dict[idx][att_info[0][1], att_info[0][0]] = att_info[1].mean(dim=1).squeeze()
    
    return att_matrix_dict


def prep_for_gatt(model, data, num_hops: int, sparse: bool = True) -> Tuple[Dict, Dict]:
    att_matrix_dict = generate_att_dict(model=model, data=data, sparse=sparse)
    correction_matrix_dict = {0: torch.eye(data.num_nodes).to_sparse().to(data.x.device)} if sparse else {0: torch.eye(data.num_nodes).to(data.x.device)}

    for idx in range(1, num_hops):
        correction_matrix_dict[idx] = torch.sparse.mm(correction_matrix_dict[idx - 1], att_matrix_dict[num_hops - idx]) if sparse else correction_matrix_dict[idx - 1] @ att_matrix_dict[num_hops - idx]

    return att_matrix_dict, correction_matrix_dict


def avgatt(target_edge: Tuple[int, int], att_matrix_dict: Dict, num_hops: int) -> float:
    return sum(att_matrix_dict[m][target_edge[1], target_edge[0]].item() for m in range(num_hops)) / num_hops


def gatt(target_edge: Tuple[int, int], ref_node: int, att_matrix_dict: Dict, correction_matrix_dict: Dict, num_hops=int) -> float:
    src_idx, tgt_idx = target_edge
    return sum(correction_matrix_dict[num_hops - m - 1][ref_node, tgt_idx].item() * att_matrix_dict[m][tgt_idx, src_idx] for m in range(num_hops - 1)) + att_matrix_dict[num_hops - 1][tgt_idx, src_idx].item()


def get_gatt(target_node, model, data, sparse: bool = True) -> Tuple[List[float], List[Tuple[int, int]]]:
    num_hops = model.num_layers
    att_matrix_dict, correction_matrix_dict = prep_for_gatt(model=model, data=data, num_hops=num_hops, sparse=sparse)

    edges_in_k_hop = return_edges_in_k_hop(data=data, target_idx=target_node, hop=num_hops, self_loops=True)
    gatt_list = [gatt(current_edge, target_node, att_matrix_dict, correction_matrix_dict, num_hops) for current_edge in edges_in_k_hop]

    return gatt_list, edges_in_k_hop


def get_avgatt(target_node, model, data, sparse: bool = True) -> Tuple[List[float], List[Tuple[int, int]]]:
    num_hops = model.num_layers
    att_matrix_dict, _ = prep_for_gatt(model=model, data=data, num_hops=num_hops, sparse=sparse)

    edges_in_k_hop = return_edges_in_k_hop(data=data, target_idx=target_node, hop=num_hops, self_loops=True)
    avgatt_list = [avgatt(current_edge, att_matrix_dict, num_hops) for current_edge in edges_in_k_hop]

    return avgatt_list, edges_in_k_hop
