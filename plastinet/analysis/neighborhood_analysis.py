import scanpy as sc
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree


def plot_top_genes_leiden(
    adata, leiden_col="leiden", top_n=10, normalize=True, custom_labels=None
):
    
    sc.tl.rank_genes_groups(adata, groupby=leiden_col)

    ranked_genes = {}
    for cluster in adata.obs[leiden_col].unique():
        try:
            ranked_genes[cluster] = adata.uns['rank_genes_groups']['names'][cluster][:top_n]
        except KeyError:
            print(f"No ranked genes available for cluster: {cluster}")
            ranked_genes[cluster] = []

    all_top_genes = list(set(gene for genes in ranked_genes.values() for gene in genes))

    expression_data = adata[:, all_top_genes].X.toarray()
    expression_df = pd.DataFrame(
        expression_data,
        index=adata.obs[leiden_col],
        columns=all_top_genes
    )
    
    grouped_expression = expression_df.groupby(expression_df.index).mean()

    if custom_labels:
        grouped_expression.index = grouped_expression.index.map(custom_labels)

    sns.clustermap(
        grouped_expression,
        figsize=(10, 10),
        cmap="coolwarm",
        standard_scale=1 if normalize else None,  
    )
    plt.title("Clustered Heatmap of Top Genes")
    plt.show()

    return ranked_genes
    
def calculate_neighborhood_colocalization(adata, x_key="X", y_key="Y", leiden_key="leiden", radius=200):
    spatial_locs = np.stack((adata.obs[x_key], adata.obs[y_key]), axis=1)
    neighborhoods = adata.obs[leiden_key].astype(str).to_numpy() 
    kd_tree = cKDTree(spatial_locs)
    unique_neighborhoods = np.unique(neighborhoods)
    
    neighbor_counts = {label: {nbr: 0 for nbr in unique_neighborhoods} for label in unique_neighborhoods}

    for cell_index, (loc, neighborhood) in enumerate(zip(spatial_locs, neighborhoods)):
        indices = kd_tree.query_ball_point(loc, radius)
        indices = [i for i in indices if i != cell_index]  
        neighbor_labels = neighborhoods[indices]

        for neighbor_label in neighbor_labels:
            neighbor_counts[neighborhood][neighbor_label] += 1

    neighbor_counts_df = pd.DataFrame(neighbor_counts).fillna(0)

    normalized_counts_df = neighbor_counts_df.div(neighbor_counts_df.sum(axis=0), axis=1)

    return normalized_counts_df

def plot_neighborhood_colocalization(colocalization_df):

    colocalization_df.T.plot(
        kind="bar", stacked=True, figsize=(12, 8), colormap="tab20", edgecolor="black"
    )

    plt.title("Neighborhood Colocalization")
    plt.xlabel("Neighborhoods")
    plt.ylabel("Proportion")
    plt.legend(title="Neighbor Neighborhoods", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
