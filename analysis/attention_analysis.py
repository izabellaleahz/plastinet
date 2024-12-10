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

