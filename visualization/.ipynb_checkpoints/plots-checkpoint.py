import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
from matplotlib import colors as mcolors


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


def plot_expression(adata, gene, x_coord='X', y_coord='Y'):
    '''
    Plot the expression level of a specific gene on the spatial map of the tissue.

    Args:
        adata: AnnData object
        gene: Gene name (must be present in `adata.var_names`)
        x_coord: Key for X-coordinate values in `adata.obs`
        y_coord: Key for Y-coordinate values in `adata.obs`
    '''
    plt.figure(figsize=(12, 8), dpi=300)

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
