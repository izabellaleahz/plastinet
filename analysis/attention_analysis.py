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
    '''
    Plot a continuous observation from the adata object.

    Args:
        adata: AnnData object
        continuous_obs_name: Name of the continuous observation in `adata.obs` to plot.
        X_key: Key for X-coordinate values in `adata.obs`
        Y_key: Key for Y-coordinate values in `adata.obs`
        size: Size of the scatter plot points
        save_path: Path to save the plot (optional). If None, displays the plot.
    '''
    plt.figure(figsize=(12, 8), dpi=300)
    ax = plt.gca()

    continuous_obs_values = adata.obs[continuous_obs_name]
    continuous_obs_values = np.ravel(continuous_obs_values)
    scatter = plt.scatter(adata.obs[X_key], adata.obs[Y_key], s=size, c=continuous_obs_values, cmap='RdGy_r')

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


def plot_attention_heatmap_for_genes(plastinet, adata, subset_key, gene_list):
    """
    Plot a heatmap of self-attention flow for a specific list of input genes.

    Parameters:
    plastinet: The model object containing self-attention weights.
    adata: Annotated data object.
    subset_key: The key in adata.obs that denotes cell types.
    gene_list: A list of genes for which the attention flow will be visualized.

    Returns:
    None, displays a heatmap of attention flow for the specified genes.
    """

    valid_genes = [gene for gene in gene_list if gene in adata.var.index]

    if len(valid_genes) == 0:
        raise ValueError("None of the genes in the input list are found in the dataset.")

    gene_indices = [np.where(adata.var.index == gene)[0][0] for gene in valid_genes]

    cell_types = set(adata.obs[subset_key])
    mean_self_attention_per_type = []
    valid_cell_types = []
    index_map = {cell: idx for idx, cell in enumerate(adata.obs.index)}

    for cell_type in cell_types:
        if isinstance(cell_type, str):
            subset_indices = adata.obs[adata.obs[subset_key] == cell_type].index
            subset_positions = [index_map[cell] for cell in subset_indices]

            attn_flow = np.sum(plastinet.self_attn_weights1[subset_positions, :], plastinet.self_attn_weights2[subset_positions, :])
            
            mean_self_attention = np.mean(attn_flow[:, gene_indices], axis=0)
            
            if mean_self_attention.size > 0:
                mean_self_attention_per_type.append(mean_self_attention)
                valid_cell_types.append(cell_type)

    mean_self_attention_matrix = np.array(mean_self_attention_per_type)

    log_mean_self_attention_matrix = np.log1p(mean_self_attention_matrix)

    attention_df = pd.DataFrame(log_mean_self_attention_matrix, index=valid_cell_types, columns=valid_genes)

    plt.figure(figsize=(10, 6))
    sns.clustermap(attention_df, cmap='RdGy_r', cbar=True, annot=True, fmt='.2f', linewidths=0.5, xticklabels=valid_genes, yticklabels=valid_cell_types)
    plt.title('Attention Flow for Selected Genes (Self-Attention)')
    plt.xlabel('Genes')
    plt.ylabel('Cell Types')
    plt.xticks(rotation=90)
    plt.show()


def analyze_attention_deg(plastinet, adata, subset_key, top_n=15, min_threshold=0.0001):
    """
    Analyze attention flow between cell types and find genes with differential attention flow.
    """
    cell_types = set(adata.obs[subset_key])
    
    mean_self_attention_per_type = []
    valid_cell_types = []
    index_map = {cell: idx for idx, cell in enumerate(adata.obs.index)}

    for cell_type in cell_types:
        if isinstance(cell_type, str):
            subset_indices = adata.obs[adata.obs[subset_key] == cell_type].index
            subset_positions = [index_map[cell] for cell in subset_indices]

            # attn_flow = np.multiply(plastinet.self_attn_weights1[subset_positions, :], plastinet.self_attn_weights2[subset_positions, :])
            attn_flow = np.add(plastinet.self_attn_weights1[subset_positions, :], plastinet.self_attn_weights2[subset_positions, :])

            mean_self_attention = np.mean(attn_flow, axis=0)

            if mean_self_attention.size > 0:
                mean_self_attention_per_type.append(mean_self_attention)
                valid_cell_types.append(cell_type)

    mean_self_attention_matrix = np.array(mean_self_attention_per_type)

    col_mask = np.nanmax(mean_self_attention_matrix, axis=0) > min_threshold
    mean_self_attention_matrix = mean_self_attention_matrix[:, col_mask]
    gene_names = adata.var.index[col_mask]

    zscored_attention_matrix = zscore(mean_self_attention_matrix, axis=0)

    attention_df = pd.DataFrame(zscored_attention_matrix, index=valid_cell_types, columns=gene_names)

    top_genes_df = pd.DataFrame(index=valid_cell_types)
    for cell_type in valid_cell_types:
        top_genes = attention_df.loc[cell_type].nlargest(top_n).index.tolist()
        top_genes_df.loc[cell_type, 'Top Genes'] = ', '.join(top_genes)

    plt.figure(figsize=(12, 8))
    sns.clustermap(zscored_attention_matrix, cmap='RdGy_r', cbar=True, row_cluster=True, col_cluster=True,
                   xticklabels=gene_names, yticklabels=valid_cell_types)
    plt.title('Z-scored Attention Flow Clustermap (Self-Attention Layer 1 * Layer 2)')
    plt.ylabel('Cell Types')
    plt.xlabel('Genes')
    plt.xticks(rotation=90)
    plt.show()

    return top_genes_df
    
def analyze_self_attention_across_pseudotime(plastinet, adata, pseudotime_key, genes_of_interest, pseudotime_bins=5):

    if pseudotime_key not in adata.obs.columns:
        raise ValueError(f"Pseudotime key '{pseudotime_key}' not found in adata.obs.")
    
    missing_genes = [gene for gene in genes_of_interest if gene not in adata.var.index]
    if missing_genes:
        raise ValueError(f"The following genes are not present in adata.var: {', '.join(missing_genes)}")
    
    gene_indices = [adata.var.index.get_loc(gene) for gene in genes_of_interest]
    
    pseudotime_vals = adata.obs[pseudotime_key].dropna()
    bins = np.linspace(pseudotime_vals.min(), pseudotime_vals.max(), pseudotime_bins + 1)
    pseudotime_labels = pd.cut(pseudotime_vals, bins=bins, labels=False)
    
    target_cells = adata.obs[~adata.obs[pseudotime_key].isna()]
    target_positions = target_cells.index.map(lambda cell: np.where(adata.obs.index == cell)[0][0])
    np.zeros((pseudotime_bins, len(genes_of_interest)))
    attention_flow_matrix = np.zeros((pseudotime_bins, len(genes_of_interest)))


    for bin_label in range(pseudotime_bins):
        bin_positions = target_cells[pseudotime_labels == bin_label].index.map(lambda cell: np.where(adata.obs.index == cell)[0][0])

        if len(bin_positions) == 0:
            continue
        
        self_attn_flows = []

       
        for target_pos in bin_positions:
           
            flow = (
                np.sum(plastinet.neighbor_attn_weights1[target_pos, :, :], axis=0)[gene_indices] +
                np.sum(plastinet.neighbor_attn_weights2[target_pos, :, :], axis=0)[gene_indices]
            )
            
            self_attn_flows.append(flow)

       
        if len(self_attn_flows) > 0:
            avg_flow = np.mean(self_attn_flows, axis=0)
            attention_flow_matrix[bin_label, :] = avg_flow

 
    gene_means = attention_flow_matrix.mean(axis=0)
    gene_stds = attention_flow_matrix.std(axis=0)
    attention_flow_matrix = (attention_flow_matrix - gene_means) / gene_stds

    plt.figure(figsize=(10, 6))
    sns.heatmap(attention_flow_matrix, cmap='RdGy_r', cbar=True, yticklabels=range(pseudotime_bins), xticklabels=genes_of_interest)
    plt.title(f'Self-Attention Flow for Specified Genes across Pseudotime')
    plt.xlabel('Genes')
    plt.ylabel('Pseudotime Bins')
    plt.show()

    attention_flow_df = pd.DataFrame(attention_flow_matrix, columns=genes_of_interest)
    
    return attention_flow_df


def visualize_gene_attention_across_pseudotime(plastinet, adata, subset_key, pseudotime_key, cell_type, genes_of_interest, pseudotime_bins=10):
    """
    Analyze attention flow across pseudotime for a specific target cell type, focusing on differential attention changes for specified genes.
    
    Parameters:
    plastinet: The model object containing attention weights.
    adata: Annotated data object.
    subset_key: The key in adata.obs that denotes cell types.
    pseudotime_key: The key in adata.obs that denotes pseudotime values.
    cell_type: The specific cell type to consider as neighbors.
    genes_of_interest: List of gene names (strings) to analyze.
    pseudotime_bins: Number of bins to group cells based on pseudotime values.

    Returns:
    A DataFrame summarizing the differential attention flow across pseudotime for each specified gene.
    """
    
    # Ensure pseudotime exists
    if pseudotime_key not in adata.obs.columns:
        raise ValueError(f"Pseudotime key '{pseudotime_key}' not found in adata.obs.")
    
    # Ensure genes_of_interest are in the adata.var index
    missing_genes = [gene for gene in genes_of_interest if gene not in adata.var.index]
    if missing_genes:
        raise ValueError(f"The following genes are not present in adata.var: {', '.join(missing_genes)}")
    
    # Map genes of interest to their indices
    gene_indices = [adata.var.index.get_loc(gene) for gene in genes_of_interest]
    
    # Get the pseudotime values and create bins
    pseudotime_vals = adata.obs[pseudotime_key].dropna()
    bins = np.linspace(pseudotime_vals.min(), pseudotime_vals.max(), pseudotime_bins + 1)
    pseudotime_labels = pd.cut(pseudotime_vals, bins=bins, labels=False)
    adata.obs["pseudotime_bins"] = pseudotime_labels  # Save pseudotime bins for plotting
    
    # Filter cells with pseudotime and map to bin
    target_cells = adata.obs[~adata.obs[pseudotime_key].isna()]
    target_positions = target_cells.index.map(lambda cell: np.where(adata.obs.index == cell)[0][0])

    # Initialize matrix for attention flow across pseudotime bins for specified genes
    attention_flow_matrix = np.zeros((pseudotime_bins, len(genes_of_interest)))

    attention_flow_results = []

    # Iterate over pseudotime bins
    for bin_label in range(pseudotime_bins):
        bin_positions = target_cells[pseudotime_labels == bin_label].index.map(lambda cell: np.where(adata.obs.index == cell)[0][0])

        if len(bin_positions) == 0:
            continue
        
        neighbor_attn_flows = []

        # For each target cell, get its neighbors and their attention flow
        for target_pos in bin_positions:
            neighbors = plastinet.neighbor_indices[target_pos]
            valid_neighbors = []

            # For each neighbor, check if it's the specified cell type
            for idx, neighbor_pos in enumerate(neighbors):
                if adata.obs.iloc[neighbor_pos][subset_key] == cell_type:
                    valid_neighbors.append(idx)

            if len(valid_neighbors) == 0:
                continue

            # Calculate attention flow from valid neighbors (specific cell type) for genes of interest
            flow = (
                np.sum(plastinet.neighbor_attn_weights1[target_pos, valid_neighbors, :][:, gene_indices], axis=0) +
                np.sum(plastinet.neighbor_attn_weights2[target_pos, valid_neighbors, :][:, gene_indices], axis=0)
            )
            
            # Normalize by the number of stromal cells in the neighborhood
            flow = flow / len(valid_neighbors) if len(valid_neighbors) > 0 else flow
            
            # Sum the flows from all neighbors for the target cell
            neighbor_attn_flows.append(flow)

        # Average the flows across all cells in the bin
        if len(neighbor_attn_flows) > 0:
            avg_flow = np.mean(neighbor_attn_flows, axis=0)
            attention_flow_matrix[bin_label, :] = avg_flow

            # Store the result for this bin
            attention_flow_results.append({
                'Pseudotime Bin': bin_label,
                'Gene Attention Flows': dict(zip(genes_of_interest, avg_flow))
            })

    # Normalize attention flow matrix with Z-score normalization across genes
    gene_means = attention_flow_matrix.mean(axis=0)
    gene_stds = attention_flow_matrix.std(axis=0)
    attention_flow_matrix = (attention_flow_matrix - gene_means) / gene_stds

    # Plot heatmap of attention flow across pseudotime bins for the specified genes
    plt.figure(figsize=(10, 6))
    sns.clustermap(attention_flow_matrix, cmap='RdGy_r', cbar=True, yticklabels=range(pseudotime_bins), xticklabels=genes_of_interest, row_cluster=False, col_cluster=True)
    plt.title(f'Differential Attention Flow from {cell_type} Neighbors across Pseudotime for Specified Genes')
    plt.xlabel('Genes')
    plt.ylabel('Pseudotime Bins')
    plt.show()
    # print(adata)
    # Plot pseudotime bins in spatial context
    plot_tissue(adata, "pseudotime_bins")
    plot_tissue(adata, "subset")

    # Ensure all gene attention values are calculated and stored
    for gene, gene_index in zip(genes_of_interest, gene_indices):
        gene_attention = np.zeros(adata.shape[0])  # Initialize attention values for all cells
        for target_pos in target_positions:
            neighbors = plastinet.neighbor_indices[target_pos]
            valid_neighbors = [idx for idx, neighbor_pos in enumerate(neighbors) if adata.obs.iloc[neighbor_pos][subset_key] == cell_type]
            if len(valid_neighbors) > 0:
                gene_attention[target_pos] = np.mean(
                    plastinet.neighbor_attn_weights1[target_pos, valid_neighbors, gene_index] *
                    plastinet.neighbor_attn_weights2[target_pos, valid_neighbors, gene_index]
                )
        adata.obs[f"{gene}_attention"] = gene_attention  # Store attention for plotting

    # Set up grid plot for attention values
    num_genes = len(genes_of_interest)
    grid_size = int(np.ceil(np.sqrt(num_genes)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()

    # Plot each gene attention in the grid
    for i, gene in enumerate(genes_of_interest):
        if f"{gene}_attention" in adata.obs:  # Check if key exists in adata.obs
            ax = axes[i]
            gene_attention_values = adata.obs[f"{gene}_attention"].values.ravel()
            
                # Scatter plot for spatial distribution of gene attention values
            scatter = ax.scatter(
                adata.obs["X"], adata.obs["Y"],
                s=1, c=gene_attention_values, cmap='RdGy_r'
            )
            
            # Color bar and title settings for each subplot
            cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f'{gene} Attention Value', rotation=270, labelpad=15)
            
            ax.set_title(f"Attention for {gene}")
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Y-axis")
        else:
            print(f"Warning: {gene}_attention not found in adata.obs, skipping this gene.")
    
    # Hide any unused subplots if the grid is larger than the number of genes
    for i in range(num_genes, grid_size * grid_size):
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.show()


    attention_flow_df = pd.DataFrame(attention_flow_matrix, columns=genes_of_interest)
    
    return attention_flow_df
