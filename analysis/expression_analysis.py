# analysis/expression_analysis.py
import scanpy as sc
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import pygam
from pygam import LinearGAM
import math
import seaborn as sns

def plot_pseudotime_heatmap(adata, gene_list, pseudotime_col='final_avg_pseudotime', n_bins=10):
   
    adata.obs[pseudotime_col] = pd.to_numeric(adata.obs[pseudotime_col], errors='coerce')
    adata = adata[~adata.obs[pseudotime_col].isna()].copy()
    
    adata.obs['pseudotime_bin'] = pd.cut(adata.obs[pseudotime_col], bins=n_bins, labels=False)
    grouped = adata.obs.groupby('pseudotime_bin')
    
    z_scored = {}
    for gene in gene_list:
        expression = []
        for bin_idx, indices in grouped.indices.items():
            mean_exp = np.mean(adata[indices, gene].X.toarray()) if len(indices) > 0 else np.nan
            expression.append(mean_exp)
        z_scores = (np.array(expression) - np.nanmean(expression)) / np.nanstd(expression)
        z_scored[gene] = z_scores
    
    z_scored_df = pd.DataFrame(z_scored, index=range(n_bins))
    
    plt.figure(figsize=(10, len(gene_list) / 2))
    sns.heatmap(
        z_scored_df,
        cmap="coolwarm",
        cbar=True,
        xticklabels=gene_list,
        yticklabels=range(n_bins),
        cbar_kws={"label": "Z-score"}
    )
    plt.gca().invert_yaxis() 
    plt.ylabel('Pseudotime Bin')
    plt.xlabel('Genes')
    plt.title('Gene Expression Heatmap Over Pseudotime')
    plt.show()
    
    return adata.obs['pseudotime_bin']


def plot_gam_curves(adata, gene_dict, pseudotime_col='final_avg_pseudotime', n_splines=5):

    adata.obs[pseudotime_col] = pd.to_numeric(adata.obs[pseudotime_col], errors='coerce')
    valid_adata = adata[~adata.obs[pseudotime_col].isna()].copy()
    pseudotime = valid_adata.obs[pseudotime_col].values

    if len(pseudotime) == 0:
        raise ValueError("Pseudotime column is empty or contains only NaN values after filtering.")

    all_genes = [gene for genes in gene_dict.values() for gene in genes]
    n_genes = len(all_genes)
    n_cols = 4  
    n_rows = math.ceil(n_genes / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), sharex=True, sharey=True)
    axes = axes.flatten() if n_genes > 1 else [axes]
    
    composite_fig, composite_ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(gene_dict))) 
    for i, (category, gene_list) in enumerate(gene_dict.items()):
        for gene in gene_list:
            
            expression = valid_adata[:, gene].X.toarray().flatten()
            
            valid_mask = np.isfinite(pseudotime) & np.isfinite(expression)
            valid_pseudotime = pseudotime[valid_mask]
            valid_expression = expression[valid_mask]
            
            if len(valid_pseudotime) < 2:  
                print(f"Not enough valid data points for gene {gene}. Skipping.")
                continue
  
            gam = LinearGAM(n_splines=n_splines).fit(valid_pseudotime, valid_expression)
            
            x = np.linspace(valid_pseudotime.min(), valid_pseudotime.max(), 100)
            y = gam.predict(x)
            
            ax_idx = all_genes.index(gene)
            ax = axes[ax_idx]
            ax.plot(x, y, label=f"{gene}")
            ax.set_title(gene, fontsize=10)
            ax.set_xlabel('Pseudotime')
            ax.set_ylabel('Expression')
            ax.legend(fontsize=8, loc="upper left")
            
            composite_ax.plot(x, y, label=f"{category} - {gene}", color=colors[i])

    for j in range(len(all_genes), len(axes)):
        axes[j].axis('off')
    
    composite_ax.set_title('Composite GAM Curves by Category', fontsize=14)
    composite_ax.set_xlabel('Pseudotime', fontsize=12)
    composite_ax.set_ylabel('Expression', fontsize=12)
    composite_ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.show()
    composite_fig.show()

# neighborhood analysis 
def plot_top_genes_leiden(adata, leiden_col='leiden', top_n=10):

    sc.tl.rank_genes_groups(adata, groupby=leiden_col)
    
    ranked_genes = {
        cluster: adata.uns['rank_genes_groups']['names'][cluster][:top_n]
        for cluster in adata.obs[leiden_col].unique()
    }
    
    all_top_genes = list(set(gene for genes in ranked_genes.values() for gene in genes))
    expression_data = adata[:, all_top_genes].X.toarray() 
    expression_df = pd.DataFrame(
        expression_data,
        index=adata.obs[leiden_col],
        columns=all_top_genes
    )
    
    sns.clustermap(
        expression_df.groupby(expression_df.index).mean(),
        figsize=(10, 10),
        cmap="coolwarm",
        standard_scale=1  # Normalize rows for better comparison
    )
    plt.title("Clustered Heatmap of Top Genes")
    plt.show()


def plot_composition_dotplot(adata, celltype_col='celltype', leiden_col='leiden'):

    composition = (
        adata.obs.groupby([leiden_col, celltype_col])
        .size()
        .unstack(fill_value=0)
        .apply(lambda x: x / x.sum(), axis=1)  
    )
    
    composition_long = composition.reset_index().melt(
        id_vars=leiden_col,
        var_name='Cell Type',
        value_name='Proportion'
    )
    
    cell_types = composition_long['Cell Type'].unique()
    palette = sns.color_palette('tab10', len(cell_types))  
    color_map = dict(zip(cell_types, palette))  
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=composition_long,
        x=leiden_col,
        y='Cell Type',
        size='Proportion',
        sizes=(75, 500),  
        hue='Cell Type', 
        palette=color_map 
    )
    
    plt.title('Relative Composition of Cell Types by Leiden Neighborhood')
    plt.xlabel('Leiden Neighborhood')
    plt.ylabel('Cell Type')
    plt.legend(title='Cell Type', loc='upper right', bbox_to_anchor=(1.2, 1), markerscale=1)
    plt.show()
