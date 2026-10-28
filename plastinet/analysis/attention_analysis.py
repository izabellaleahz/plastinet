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

def analyze_self_attention_layer(
    embedding_adata, adata, cell_type_col='subset', gene_list=None, normalize=True, layer = "1", top_n_genes=20
):
   
    # Drop NaN values in cell type column (before reading weights so row positions stay aligned)
    embedding_adata = embedding_adata[~embedding_adata.obs[cell_type_col].isna()].copy()
    attention_weights = embedding_adata.obsm['self_attention_weights_layer' + layer]
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
        attention_df = attention_df.apply(zscore, axis=1, result_type="broadcast")

    # Visualization
    plt.figure(figsize=(12, 8))
    sns.heatmap(attention_df, cmap='coolwarm', annot=False, fmt='.2f', linewidths=0.5)
    plt.title(f"Self-Attention Patterns (Layer 1) - {'Z-Scored' if normalize else 'Raw'}")
    plt.xlabel("Cell Types")
    plt.ylabel("Genes")
    plt.show()
    
def analyze_attention_flow(
    embedding_adata, adata, cell_type_col='subset', gene_list=None, normalize=True):
    embedding_adata = embedding_adata[~embedding_adata.obs[cell_type_col].isna()].copy()

    layer1_weights = embedding_adata.obsm['self_attention_weights_layer1']
    layer2_weights = embedding_adata.obsm['self_attention_weights_layer2']

    attention_flow = layer1_weights * layer2_weights

    cell_types = embedding_adata.obs[cell_type_col].unique()

    gene_list = [gene for gene in gene_list if gene in adata.var.index]
        
    gene_indices = [adata.var.index.get_loc(gene) for gene in gene_list]

    mean_attention_flow = {}
    for cell_type in cell_types:
        cell_indices = embedding_adata.obs[cell_type_col] == cell_type
        cell_indices = cell_indices.values.nonzero()[0]

        if len(cell_indices) == 0:
            print(f"No cells found for cell type {cell_type}. Skipping.")
            continue

        mean_attention_flow[cell_type] = attention_flow[cell_indices][:, gene_indices].mean(axis=0)

    if not mean_attention_flow:
        raise ValueError("No valid cell types or genes were found for analysis.")

    attention_df = pd.DataFrame(mean_attention_flow, index=gene_list)

    if normalize:
        attention_df = attention_df.apply(zscore, axis=1, result_type="broadcast")

    plt.figure(figsize=(12, 8))
    sns.heatmap(attention_df, cmap='coolwarm', annot=False, fmt='.2f', linewidths=0.5)
    plt.title(f"Attention Flow (Layer 1 * Layer 2) - {'Z-Scored' if normalize else 'Raw'}")
    plt.xlabel("Cell Types")
    plt.ylabel("Genes")
    plt.show()

    return attention_df


from scipy.stats import ttest_1samp

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
def analyze_attention_flow_with_permutation(
    embedding_adata, adata, cell_type_col='subset', gene_list=None, n_permutations=1000, random_seed=0
):
    # Filter valid cells before reading weights so row positions stay aligned
    embedding_adata = embedding_adata[~embedding_adata.obs[cell_type_col].isna()].copy()
    cell_types = embedding_adata.obs[cell_type_col].unique()

    # Extract attention weights
    layer1_weights = embedding_adata.obsm['self_attention_weights_layer1']
    layer2_weights = embedding_adata.obsm['self_attention_weights_layer2']

    # Compute attention flow (element-wise multiplication)
    attention_flow = layer1_weights * layer2_weights

    gene_list = [gene for gene in gene_list if gene in adata.var.index]
    gene_indices = [adata.var.index.get_loc(gene) for gene in gene_list]

    flow_genes = attention_flow[:, gene_indices]
    overall_mean = flow_genes.mean(axis=0)
    n_cells = flow_genes.shape[0]
    rng = np.random.default_rng(random_seed)

    # Compute mean attention flow and p-values
    mean_attention_flow = {}
    p_values = {}
    for cell_type in cell_types:
        cell_indices = embedding_adata.obs[cell_type_col] == cell_type
        cell_indices = cell_indices.values.nonzero()[0]

        if len(cell_indices) == 0:
            print(f"No cells found for cell type {cell_type}. Skipping.")
            continue

        mean_flow = flow_genes[cell_indices].mean(axis=0)
        mean_attention_flow[cell_type] = mean_flow

        # Null: mean over a random set of the same size drawn from all cells (label permutation)
        permuted_means = np.array([
            flow_genes[rng.choice(n_cells, size=len(cell_indices), replace=False)].mean(axis=0)
            for _ in range(n_permutations)
        ])

        # Two-sided p-value on the deviation from the all-cell mean, with the +1 correction
        observed_dev = np.abs(mean_flow - overall_mean)
        null_dev = np.abs(permuted_means - overall_mean)
        p_values[cell_type] = (1 + np.sum(null_dev >= observed_dev, axis=0)) / (1 + n_permutations)

    attention_df = pd.DataFrame(mean_attention_flow, index=gene_list)
    pval_df = pd.DataFrame(p_values, index=gene_list)

    # Mask non-significant genes (p-value > 0.05)
    significant_mask = pval_df <= 0.05  # Boolean mask for significant genes
    
    # Create a masked DataFrame for visualization
    masked_attention_df = attention_df.where(significant_mask, other=np.nan)  # Replace non-significant values with NaN
    
    # Heatmap of significant genes only
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        masked_attention_df, 
        cmap="coolwarm", 
        annot=False, 
        fmt=".2f", 
        linewidths=0.5, 
        mask=~significant_mask  # Mask for non-significant genes
    )
    plt.title("Significant Attention Flow (Layer 1 * Layer 2)")
    plt.xlabel("Cell Types")
    plt.ylabel("Genes")
    plt.show()


    return pval_df



def analyze_stromal_to_epi_attention(
    embedding_adata,
    adata,
    pseudotime_key: str,
    cell_type_col: str = "subset",
    source_type: str = "stromal",
    target_type: str = "epi",
    gene_list: list = None,
    normalize: bool = True,
    top_n_genes: int = 20
):
    """
    Analyze stromal-to-epithelial neighbor attention for specific genes across a pseudotime gradient.

    Parameters:
    - embedding_adata: AnnData object containing neighbor attention weights and pseudotime data.
    - adata: AnnData object with gene information in `.var` to align attention weights with genes.
    - pseudotime_key: Key in `embedding_adata.obs` for pseudotime values.
    - cell_type_col: Column in `embedding_adata.obs` defining cell types.
    - source_type: The source cell type (e.g., "stromal").
    - target_type: The target cell type (e.g., "epi").
    - gene_list: List of genes to analyze (optional). If None, selects top genes by variance.
    - normalize: Whether to apply Z-score normalization (default: True).
    - top_n_genes: Number of genes to select if `gene_list` is not provided.

    Returns:
    - DataFrame summarizing neighbor attention across pseudotime.
    """
    # Neighbor attention edge list: attention that cell `target` pays to neighbor `source` (row positions)
    neighbor_attention = embedding_adata.uns["neighbor_attention"]
    targets = np.asarray(neighbor_attention["target"])
    sources = np.asarray(neighbor_attention["source"])
    attention_layer1 = np.asarray(neighbor_attention["layer1"])

    if targets.size and max(targets.max(), sources.max()) >= embedding_adata.n_obs:
        raise ValueError("Neighbor attention indices do not match embedding_adata; pass the unsubsetted embedding_adata.")

    cell_types = embedding_adata.obs[cell_type_col].to_numpy()
    if not np.any(cell_types == source_type):
        raise ValueError(f"No cells found for source type: {source_type}")
    if not np.any(cell_types == target_type):
        raise ValueError(f"No cells found for target type: {target_type}")

    # Edges from a source-type neighbor into a target-type cell
    edge_mask = (cell_types[targets] == target_type) & (cell_types[sources] == source_type)
    if not np.any(edge_mask):
        raise ValueError(f"No {target_type} cells have {source_type} neighbors.")

    # Handle gene selection
    if gene_list is None:
        # Compute variance of attention weights across all genes
        gene_variances = attention_layer1[edge_mask].var(axis=0)
        top_gene_indices = np.argsort(gene_variances)[-top_n_genes:]  # Select top N genes by variance
        gene_list = [adata.var.index[i] for i in top_gene_indices]
    else:
        # Validate provided gene list
        gene_list = [gene for gene in gene_list if gene in adata.var.index]
        if not gene_list:
            raise ValueError("None of the specified genes are present in adata.var.")

    # Map gene list to indices
    gene_indices = [adata.var.index.get_loc(gene) for gene in gene_list]

    # Mean attention to source-type neighbors per target cell; cells without such neighbors are left out
    edge_df = pd.DataFrame(attention_layer1[edge_mask][:, gene_indices], columns=gene_list)
    edge_df["target"] = targets[edge_mask]
    per_cell = edge_df.groupby("target").mean()

    pseudotime = pd.to_numeric(embedding_adata.obs[pseudotime_key], errors="coerce").to_numpy()[per_cell.index]
    has_pseudotime = ~np.isnan(pseudotime)
    per_cell, pseudotime = per_cell[has_pseudotime], pseudotime[has_pseudotime]
    order = np.argsort(pseudotime, kind="stable")

    n_target = int(np.sum(cell_types == target_type))
    print(f"{len(per_cell)} of {n_target} {target_type} cells have {source_type} neighbors and a pseudotime value.")

    # Convert to DataFrame
    attention_df = pd.DataFrame(
        per_cell.to_numpy()[order], columns=gene_list, index=pseudotime[order]
    )

    # Apply Z-score normalization if specified
    if normalize:
        attention_df = attention_df.apply(zscore, axis=0, result_type="broadcast")

    # Visualization
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        attention_df.T,
        cmap="coolwarm",
        annot=False,
        fmt=".2f",
        linewidths=0.5,
        xticklabels=False,
    )
    plt.title(f"Neighbor Attention: {source_type} to {target_type} Across Pseudotime")
    plt.xlabel("Pseudotime")
    plt.ylabel("Genes")
    plt.show()

    return attention_df
