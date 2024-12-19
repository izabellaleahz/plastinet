# plastinet/analysis/attention_analysis.py
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Dict
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, add_self_loops, k_hop_subgraph
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

def analyze_self_attention_layer(
    embedding_adata, adata, cell_type_col='subset', gene_list=None, normalize=True, layer="1", top_n_genes=20
):
    # Load the attention weights
    attention_key = 'self_attention_weights_layer' + layer
    if attention_key not in embedding_adata.obsm:
        raise ValueError(f"{attention_key} not found in embedding_adata.obsm")

    attention_weights = embedding_adata.obsm[attention_key]

    # Remove rows that are clearly invalid (e.g., padded rows that ended up as all ones)
    # You can adjust this criterion if necessary.
    invalid_mask = np.all(attention_weights == 1.0, axis=1)
    if np.any(invalid_mask):
        print(f"Found {invalid_mask.sum()} invalid rows in attention_weights. Removing them.")
        attention_weights = attention_weights[~invalid_mask]
        embedding_adata = embedding_adata[~invalid_mask].copy()

    # Filter out cells with missing cell_type_col
    embedding_adata = embedding_adata[~embedding_adata.obs[cell_type_col].isna()].copy()
    cell_types = embedding_adata.obs[cell_type_col].unique()

    # If no gene_list is provided, select top variable genes based on attention variance
    if gene_list is None:
        gene_variances = attention_weights.var(axis=0)
        top_gene_indices = np.argsort(gene_variances)[-top_n_genes:]
        gene_list = [adata.var.index[i] for i in top_gene_indices if i < len(adata.var.index)]
        print(f"Automatically selected top {len(gene_list)} genes based on variance.")
    else:
        # Validate gene_list
        gene_list = [gene for gene in gene_list if gene in adata.var.index]
        if not gene_list:
            raise ValueError("None of the specified genes are present in adata.var.")

    gene_indices = [adata.var.index.get_loc(gene) for gene in gene_list]

    mean_attention = {}
    for cell_type in cell_types:
        cell_indices = embedding_adata.obs[cell_type_col] == cell_type
        cell_indices = cell_indices.values.nonzero()[0]

        if len(cell_indices) == 0:
            print(f"No cells found for cell type {cell_type}. Skipping.")
            continue

        # Compute mean attention for the selected genes in this cell type
        mean_attention[cell_type] = attention_weights[cell_indices][:, gene_indices].mean(axis=0)

    if not mean_attention:
        raise ValueError("No valid cell types or genes were found for analysis.")

    attention_df = pd.DataFrame(mean_attention, index=gene_list)

    # Optionally normalize by gene (z-score)
    if normalize:
        attention_df = attention_df.apply(zscore, axis=1)

    plt.figure(figsize=(12, 8))
    sns.heatmap(attention_df, cmap='coolwarm', annot=False, fmt='.2f', linewidths=0.5)
    plt.title(f"Self-Attention Patterns (Layer {layer}) - {'Z-Scored' if normalize else 'Raw'}")
    plt.xlabel("Cell Types")
    plt.ylabel("Genes")
    plt.show()


def analyze_attention_flow(
    embedding_adata, adata, cell_type_col='subset', gene_list=None, normalize=True
):
    # Check keys
    if 'self_attention_weights_layer1' not in embedding_adata.obsm or \
       'self_attention_weights_layer2' not in embedding_adata.obsm:
        raise ValueError("Required keys not found in embedding_adata.obsm")

    layer1_weights = embedding_adata.obsm['self_attention_weights_layer1']
    layer2_weights = embedding_adata.obsm['self_attention_weights_layer2']

    # Filter invalid rows (all ones) if necessary
    invalid_mask_1 = np.all(layer1_weights == 1.0, axis=1)
    invalid_mask_2 = np.all(layer2_weights == 1.0, axis=1)
    invalid_mask = invalid_mask_1 | invalid_mask_2
    if np.any(invalid_mask):
        print(f"Found {invalid_mask.sum()} invalid rows. Removing them.")
        layer1_weights = layer1_weights[~invalid_mask]
        layer2_weights = layer2_weights[~invalid_mask]
        embedding_adata = embedding_adata[~invalid_mask].copy()

    # Compute attention flow
    attention_flow = layer1_weights * layer2_weights

    embedding_adata = embedding_adata[~embedding_adata.obs[cell_type_col].isna()].copy()
    cell_types = embedding_adata.obs[cell_type_col].unique()

    # Validate gene_list
    if gene_list is None:
        # Select all genes
        gene_list = list(adata.var.index)
    else:
        gene_list = [gene for gene in gene_list if gene in adata.var.index]

    if not gene_list:
        raise ValueError("No valid genes found.")

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
        attention_df = attention_df.apply(zscore, axis=1)

    plt.figure(figsize=(12, 8))
    sns.heatmap(attention_df, cmap='coolwarm', annot=False, fmt='.2f', linewidths=0.5)
    plt.title(f"Attention Flow (Layer 1 * Layer 2) - {'Z-Scored' if normalize else 'Raw'}")
    plt.xlabel("Cell Types")
    plt.ylabel("Genes")
    plt.show()

    return attention_df


def analyze_attention_flow_with_permutation(
    embedding_adata, adata, cell_type_col='subset', gene_list=None, n_permutations=1000
):
    if 'self_attention_weights_layer1' not in embedding_adata.obsm or \
       'self_attention_weights_layer2' not in embedding_adata.obsm:
        raise ValueError("Required keys not found in embedding_adata.obsm")

    layer1_weights = embedding_adata.obsm['self_attention_weights_layer1']
    layer2_weights = embedding_adata.obsm['self_attention_weights_layer2']

    # Filter invalid rows
    invalid_mask_1 = np.all(layer1_weights == 1.0, axis=1)
    invalid_mask_2 = np.all(layer2_weights == 1.0, axis=1)
    invalid_mask = invalid_mask_1 | invalid_mask_2
    if np.any(invalid_mask):
        print(f"Found {invalid_mask.sum()} invalid rows. Removing them.")
        layer1_weights = layer1_weights[~invalid_mask]
        layer2_weights = layer2_weights[~invalid_mask]
        embedding_adata = embedding_adata[~invalid_mask].copy()

    # Compute attention flow
    attention_flow = layer1_weights * layer2_weights

    embedding_adata = embedding_adata[~embedding_adata.obs[cell_type_col].isna()].copy()
    cell_types = embedding_adata.obs[cell_type_col].unique()

    if gene_list is None:
        gene_list = list(adata.var.index)
    else:
        gene_list = [gene for gene in gene_list if gene in adata.var.index]

    if not gene_list:
        raise ValueError("No valid genes found.")

    gene_indices = [adata.var.index.get_loc(gene) for gene in gene_list]

    mean_attention_flow = {}
    p_values = {}
    for cell_type in cell_types:
        cell_indices = embedding_adata.obs[cell_type_col] == cell_type
        cell_indices = cell_indices.values.nonzero()[0]

        if len(cell_indices) == 0:
            print(f"No cells found for cell type {cell_type}. Skipping.")
            continue

        mean_flow = attention_flow[cell_indices][:, gene_indices].mean(axis=0)
        mean_attention_flow[cell_type] = mean_flow

        permuted_means = []
        for _ in range(n_permutations):
            shuffled_indices = np.random.permutation(cell_indices)
            permuted_mean = attention_flow[shuffled_indices][:, gene_indices].mean(axis=0)
            permuted_means.append(permuted_mean)
        permuted_means = np.array(permuted_means)

        # Calculate p-values
        p_values[cell_type] = np.mean(np.abs(permuted_means) >= np.abs(mean_flow), axis=0)

    attention_df = pd.DataFrame(mean_attention_flow, index=gene_list)
    pval_df = pd.DataFrame(p_values, index=gene_list)

    # Mask non-significant genes
    significant_mask = pval_df <= 0.05
    masked_attention_df = attention_df.where(significant_mask, other=np.nan)

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        masked_attention_df, 
        cmap="coolwarm", 
        annot=False, 
        fmt=".2f", 
        linewidths=0.5, 
        mask=~significant_mask
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

    # Check required keys in uns
    if "neighbor_attention" not in embedding_adata.uns or \
       "layer1" not in embedding_adata.uns["neighbor_attention"] or \
       "indices" not in embedding_adata.uns["neighbor_attention"]:
        raise ValueError("neighbor_attention keys not found in embedding_adata.uns")

    neighbor_attention_layer1 = embedding_adata.uns["neighbor_attention"]["layer1"]
    neighbor_indices = embedding_adata.uns["neighbor_attention"]["indices"]

    # Convert to arrays and filter out invalid entries if needed
    neighbor_attention_layer1 = [arr for arr in neighbor_attention_layer1 if not np.all(arr == 1.0)]

    # Get cell indices for source and target
    source_indices = embedding_adata.obs[cell_type_col] == source_type
    target_indices = embedding_adata.obs[cell_type_col] == target_type

    if not np.any(source_indices):
        raise ValueError(f"No cells found for source type: {source_type}")
    if not np.any(target_indices):
        raise ValueError(f"No cells found for target type: {target_type}")

    # Extract pseudotime and sort target cells by pseudotime
    pseudotime = embedding_adata.obs.loc[target_indices, pseudotime_key]
    sorted_target_indices = pseudotime.sort_values().index

    # Handle gene selection
    if gene_list is None:
        # Compute variance of attention weights across all genes from neighbor_attention_layer1
        # Flatten them
        all_attention = np.vstack(neighbor_attention_layer1)
        gene_variances = all_attention.var(axis=0)
        top_gene_indices = np.argsort(gene_variances)[-top_n_genes:]
        gene_list = [adata.var.index[i] for i in top_gene_indices]
    else:
        gene_list = [gene for gene in gene_list if gene in adata.var.index]
        if not gene_list:
            raise ValueError("None of the specified genes are present in adata.var.")

    gene_indices = [adata.var.index.get_loc(gene) for gene in gene_list]

    attention_values = []
    for target_idx in sorted_target_indices:
        int_idx = embedding_adata.obs.index.get_loc(target_idx)

        if int_idx >= len(neighbor_attention_layer1) or int_idx >= len(neighbor_indices):
            print(f"Skipping invalid int_idx: {int_idx}")
            continue

        # Access neighbor indices and attention weights
        n_indices = neighbor_indices[int_idx]
        if int_idx >= len(neighbor_attention_layer1):
            print(f"No neighbor attention data for int_idx: {int_idx}")
            continue

        attention_weights = neighbor_attention_layer1[int_idx]

        # Filter for source cells
        valid_source_mask = embedding_adata.obs.index.isin(n_indices) & source_indices
        if np.any(valid_source_mask):
            # Extract only source neighbor attention
            # We need to match indices. If neighbor_indices are cell IDs, map them:
            valid_source_ids = embedding_adata.obs[valid_source_mask].index
            src_mask = np.isin(n_indices, valid_source_ids)
            filtered_attention = attention_weights[src_mask, :]
            attention_values.append(filtered_attention.mean(axis=0))
        else:
            attention_values.append(np.zeros(len(gene_list)))

    attention_df = pd.DataFrame(
        attention_values, columns=gene_list, index=pseudotime.loc[sorted_target_indices].values
    )

    if normalize:
        attention_df = attention_df.apply(zscore, axis=0, result_type="broadcast")

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
