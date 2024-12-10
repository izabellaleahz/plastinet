import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
from matplotlib import colors as mcolors
import networkx as nx
from torch_geometric.utils import to_networkx
<<<<<<< HEAD
from matplotlib.patches import Circle
import random

def plot_continous_obs(adata, continuous_obs_name, X_key="X", Y_key="Y", size=1, figure_size=(10, 8), save_path=None):
=======

def plot_continous_obs(adata, continuous_obs_name, X_key="X", Y_key="Y", size=1, save_path=None):
>>>>>>> e2126d572fe3fd096e14f36fc038f7141668dfe2
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
    plt.figure(figsize=figure_size, dpi=300)
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


def plot_tissue(adata, leiden_key, x_coord='X', y_coord='Y', size=1, tabTen=True, figure_size=(10, 8), save_path=None):
    '''
    Plot the tissue clusters using Leiden clustering or other categorical obs.

    Args:
        adata: AnnData object
        leiden_key: Key in `adata.obs` representing cluster labels
        x_coord: Key for X-coordinate values in `adata.obs`
        y_coord: Key for Y-coordinate values in `adata.obs`
        size: Size of the scatter plot points
        tabTen: Boolean for whether to use the 'tab10' colormap (default). If False, uses 'tab20'.
        figure_size: Tuple for figure size
        save_path: Path to save the plot (optional). If None, displays the plot.
    '''
    if leiden_key not in adata.obs:
        raise ValueError(f"{leiden_key} not found in adata.obs")

    plt.figure(figsize=figure_size, dpi=300)
    ax = plt.gca()
    cmap = plt.get_cmap('tab10' if tabTen else 'tab20')

    unique_clusters = sorted(adata.obs[leiden_key].unique(), key=str)
    color_mapping = {cluster: cmap(i % cmap.N) for i, cluster in enumerate(unique_clusters)}

    legend_handles = []
    for cluster in unique_clusters:
        subset = adata[adata.obs[leiden_key] == cluster]
        color = color_mapping[cluster]
        scatter = plt.scatter(subset.obs[x_coord], subset.obs[y_coord], label=f'Cluster {cluster}', color=color, s=size)

        legend_handle = mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=10, label=f'Cluster {cluster}')
        legend_handles.append(legend_handle)

    plt.title(f"Clusters: {leiden_key} in Tissue")
    plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel(x_coord)
    plt.ylabel(y_coord)
    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    return


def plot_composition_dot_plot(adata, cluster_key, obs_key):
    '''
    Plot the composition of a cluster by an observation key, normalized by the obs key across clusters.

    Args:
        adata: AnnData object
        cluster_key: Key in `adata.obs` representing cluster labels
        obs_key: Key in `adata.obs` representing cell types or other categorical metadata
    '''
    subtype_counts = adata.obs.groupby([cluster_key, obs_key]).size().unstack(fill_value=0)
    total_counts_per_subtype = subtype_counts.sum(axis=0)
    normalized_subtype_counts = subtype_counts.div(total_counts_per_subtype, axis=1)

    flattened_data = normalized_subtype_counts.reset_index().melt(id_vars=cluster_key, value_name='frequency', var_name=obs_key)
    fig, ax = plt.subplots(figsize=(10, 6))

    leiden = flattened_data[cluster_key].unique()
    celltypes = flattened_data[obs_key].unique()

    for subtype in celltypes:
        data_subset = flattened_data[flattened_data[obs_key] == subtype]
        ax.scatter(data_subset[cluster_key], [subtype] * len(data_subset), s=data_subset['frequency'] * 1000, alpha=0.7)

    ax.set_xlabel(cluster_key)
    ax.set_ylabel(obs_key)
    ax.set_title(f'{obs_key} vs {cluster_key}')
    plt.xticks(leiden, rotation=45)
    plt.yticks(celltypes)

    plt.show()
    return


def plot_expression(adata, gene, x_coord='X', y_coord='Y', figure_size=(12, 8)):
    '''
    Plot the expression level of a specific gene on the spatial map of the tissue.

    Args:
        adata: AnnData object
        gene: Gene name (must be present in `adata.var_names`)
        x_coord: Key for X-coordinate values in `adata.obs`
        y_coord: Key for Y-coordinate values in `adata.obs`
    '''
    plt.figure(figsize=figure_size, dpi=300)

    if gene not in adata.var_names:
        raise ValueError(f"{gene} not found in adata.var_names.")

    gene_expression = adata[:, gene].X
    if hasattr(gene_expression, "toarray"):
        gene_expression = gene_expression.toarray().flatten()

    scatter = plt.scatter(adata.obs[x_coord], adata.obs[y_coord], c=gene_expression, s=1, cmap='RdGy_r')
    plt.colorbar(scatter, label=f'Expression level of {gene}')
    plt.title(f'Expression of {gene} in Tissue')
    plt.xlabel(x_coord)
    plt.ylabel(y_coord)
    plt.tight_layout()
    plt.show()
    return

def plot_graph(graph, node_color='blue', edge_color='gray', node_size=50, figsize=(10, 10)):
    """
    Visualize the graph with nodes and edges using networkx and matplotlib.

    Plot using network graphing functions to show how inner connected the network is - so nodes and edges as defined in the constructor of the plastinet class

    should only be a couple lines 

    Eventually want to modify this to show the line weight based on the distance decay - in the attention.py file there is a defined alpha and expoential decay so the line weight representing that is goal eventually so user can pick appropiate values 
    
    """ 
    #TODO 
    return 
<<<<<<< HEAD

def plot_subsample_radius(adata, x_samples=10, radius=50, x_key='X', y_key='Y'):

    x_coords = adata.obs[x_key].values
    y_coords = adata.obs[y_key].values
    cell_positions = np.column_stack((x_coords, y_coords))
    
    sampled_indices = random.sample(range(len(cell_positions)), x_samples)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    cells_in_radius = []

    for idx in sampled_indices:
        cell_x, cell_y = cell_positions[idx]
        
        distances = np.sqrt((cell_positions[:, 0] - cell_x) ** 2 + (cell_positions[:, 1] - cell_y) ** 2)
        
        count_in_radius = np.sum(distances <= radius)
        cells_in_radius.append(count_in_radius)

        ax[0].scatter(cell_x, cell_y, color='blue', label='Sampled cell' if idx == sampled_indices[0] else "")
        circle = Circle((cell_x, cell_y), radius, color='red', alpha=0.3, linestyle='--')
        ax[0].add_patch(circle)

    ax[0].scatter(x_coords, y_coords, color='gray', alpha=0.5, s=5, label='All cells')
    ax[0].set_title('Subsampled Cells with Radius')
    ax[0].set_xlabel('X Position')
    ax[0].set_ylabel('Y Position')
    ax[0].legend()

    ax[1].hist(cells_in_radius, bins=10, color='blue', edgecolor='black')
    ax[1].set_title('Histogram of Cell Counts within Radius')
    ax[1].set_xlabel('Number of Cells within Radius')
    ax[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()
=======
>>>>>>> e2126d572fe3fd096e14f36fc038f7141668dfe2
