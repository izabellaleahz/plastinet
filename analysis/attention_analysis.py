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
<<<<<<< HEAD
from scipy.stats import zscore

from plastinet.visualization.plots import plot_tissue

def plot_continous_obs(adata, continuous_obs_name, X_key="X", Y_key="Y", size=1, save_path=None):

    plt.figure(figsize=(12, 8), dpi=300)
    ax = plt.gca()

    continuous_obs_values = adata.obs[continuous_obs_name]
    continuous_obs_values = np.ravel(continuous_obs_values)
    scatter = plt.scatter(adata.obs[X_key], adata.obs[Y_key], s=size, c=continuous_obs_values, cmap='coolwarm')

    cbar = plt.colorbar(scatter)
    cbar.set_label(f'Value of {continuous_obs_name}')

    plt.title(f"{continuous_obs_name}")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    return

import numpy as np

def analyze_self_attention_layer1(
    embedding_adata, adata, cell_type_col='subset', gene_list=None, normalize=True, top_n_genes=20
):
    """
    Analyze self-attention (layer 1) by cell type using `embedding_adata` for attention weights
    and `adata` for gene information, with automatic gene selection if no gene list is provided.

    Parameters:
    - embedding_adata: AnnData object containing self-attention weights in `obsm`.
    - adata: AnnData object containing gene information in `var`.
    - cell_type_col: Column in `embedding_adata.obs` defining cell types.
    - gene_list: List of genes to analyze (optional).
    - normalize: Whether to apply Z-score normalization (default: True).
    - top_n_genes: Number of genes to select if `gene_list` is not provided.

    Returns:
    - DataFrame summarizing mean self-attention by cell type and gene.
    """
    attention_weights = embedding_adata.obsm['self_attention_weights_layer1']

    # Drop NaN values in cell type column
    embedding_adata = embedding_adata[~embedding_adata.obs[cell_type_col].isna()].copy()
    cell_types = embedding_adata.obs[cell_type_col].unique()

    # Handle gene selection
    if gene_list is None:
        # Compute variance of attention weights across all genes
        gene_variances = attention_weights.var(axis=0)
        top_gene_indices = np.argsort(gene_variances)[-top_n_genes:]  # Select top N genes by variance
        gene_list = [adata.var.index[i] for i in top_gene_indices if i < len(adata.var.index)]
        print(f"Automatically selected top {len(gene_list)} genes based on variance.")
    else:
        # Validate provided gene list
        gene_list = [gene for gene in gene_list if gene in adata.var.index]
        if not gene_list:
            raise ValueError("None of the specified genes are present in adata.var.")

    gene_indices = [adata.var.index.get_loc(gene) for gene in gene_list]

    # Calculate mean attention
    mean_attention = {}
    for cell_type in cell_types:
        # Get indices for cells of this type in embedding_adata
        cell_indices = embedding_adata.obs[cell_type_col] == cell_type
        cell_indices = cell_indices.values.nonzero()[0]

        if len(cell_indices) == 0:
            print(f"No cells found for cell type {cell_type}. Skipping.")
            continue

        # Compute mean attention for the genes
        mean_attention[cell_type] = attention_weights[cell_indices][:, gene_indices].mean(axis=0)

    if not mean_attention:
        raise ValueError("No valid cell types or genes were found for analysis.")

    # Map gene indices to names using adata
    attention_df = pd.DataFrame(mean_attention, index=gene_list)

    # Apply Z-score normalization if specified
    if normalize:
        attention_df = attention_df.apply(zscore, axis=1)

    # Visualization
    plt.figure(figsize=(12, 8))
    sns.heatmap(attention_df, cmap='coolwarm', annot=False, fmt='.2f', linewidths=0.5)
    plt.title(f"Self-Attention Patterns (Layer 1) - {'Z-Scored' if normalize else 'Raw'}")
    plt.xlabel("Cell Types")
    plt.ylabel("Genes")
    plt.show()

    return attention_df

=======

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
>>>>>>> e2126d572fe3fd096e14f36fc038f7141668dfe2
