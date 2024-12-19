import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pygam
from pygam import LinearGAM
import math
import seaborn as sns


def construct_differentiation_path(embedding_adata, adata, cell_type_obs, cell_type, starting_point_gene_list, end_point_gene_list=None, N=5):
    gat_epi = embedding_adata[embedding_adata.obs[cell_type_obs] == cell_type]
    exp_epi = adata[adata.obs[cell_type_obs] == cell_type]

    pseudotime_df = pd.DataFrame(index=gat_epi.obs_names)

    # Score starting points
    sc.tl.score_genes(exp_epi, starting_point_gene_list, score_name='starting_score')
    gat_epi.obs['starting_score'] = exp_epi.obs['starting_score']
    top_cells_indices = gat_epi.obs['starting_score'].nlargest(N).index

    sc.pp.neighbors(gat_epi, use_rep='X')
    sub_adata = gat_epi.copy()

    # Compute pseudotime starting from top cells
    for idx, top_cell in enumerate(top_cells_indices, start=1):
        sub_adata.uns['iroot'] = np.flatnonzero(sub_adata.obs_names == top_cell)[0]
        sc.tl.dpt(sub_adata, n_branchings=0)

        pseudotime_key = f'dpt_pseudotime_global_{idx}'
        pseudotime_df[pseudotime_key] = sub_adata.obs['dpt_pseudotime'].reindex(pseudotime_df.index)

    classical_keys = [f'dpt_pseudotime_global_{i}' for i in range(1, N + 1)]
    pseudotime_df['avg_start_pseudotime'] = pseudotime_df[classical_keys].mean(axis=1, skipna=True)

    if end_point_gene_list is None:
        # Invert starting scores for ending points
        exp_epi.obs['neg_starting_score'] = -exp_epi.obs['starting_score']
        gat_epi.obs['neg_starting_score'] = exp_epi.obs['neg_starting_score']
        top_negative_indices = gat_epi.obs['neg_starting_score'].nlargest(N).index

        for idx, top_cell in enumerate(top_negative_indices, start=1):
            sub_adata.uns['iroot'] = np.flatnonzero(sub_adata.obs_names == top_cell)[0]
            sc.tl.dpt(sub_adata, n_branchings=0)

            max_pseudotime = sub_adata.obs['dpt_pseudotime'].max()
            inverted_pseudotime_key = f'inverted_dpt_pseudotime_global_{idx}'
            pseudotime_df[inverted_pseudotime_key] = max_pseudotime - sub_adata.obs['dpt_pseudotime'].reindex(pseudotime_df.index)

        inverted_negative_keys = [f'inverted_dpt_pseudotime_global_{i}' for i in range(1, N + 1)]
        pseudotime_df['avg_inverted_neg_pseudotime'] = pseudotime_df[inverted_negative_keys].mean(axis=1, skipna=True)

        pseudotime_df['final_avg_pseudotime'] = pseudotime_df[['avg_start_pseudotime', 'avg_inverted_neg_pseudotime']].mean(axis=1, skipna=True)

    else:
        # Score ending points explicitly
        sc.tl.score_genes(exp_epi, end_point_gene_list, score_name='ending_score')
        gat_epi.obs['ending_score'] = exp_epi.obs['ending_score']
        top_basal_indices = gat_epi.obs['ending_score'].nlargest(N).index

        for idx, top_cell in enumerate(top_basal_indices, start=1):
            sub_adata.uns['iroot'] = np.flatnonzero(sub_adata.obs_names == top_cell)[0]
            sc.tl.dpt(sub_adata, n_branchings=0)

            max_pseudotime = sub_adata.obs['dpt_pseudotime'].max()
            inverted_pseudotime_key = f'inverted_dpt_pseudotime_global_{idx}'
            pseudotime_df[inverted_pseudotime_key] = max_pseudotime - sub_adata.obs['dpt_pseudotime'].reindex(pseudotime_df.index)

        inverted_basal_keys = [f'inverted_dpt_pseudotime_global_{i}' for i in range(1, N + 1)]
        pseudotime_df['avg_inverted_end_pseudotime'] = pseudotime_df[inverted_basal_keys].mean(axis=1, skipna=True)

        pseudotime_df['final_avg_pseudotime'] = pseudotime_df[['avg_start_pseudotime', 'avg_inverted_end_pseudotime']].mean(axis=1, skipna=True)

    # Normalize pseudotime
    if 'final_avg_pseudotime' in pseudotime_df:
        pseudotime_df['final_avg_pseudotime'] = (
            pseudotime_df['final_avg_pseudotime'] - pseudotime_df['final_avg_pseudotime'].min()
        ) / (
            pseudotime_df['final_avg_pseudotime'].max() - pseudotime_df['final_avg_pseudotime'].min()
        )
    else:
        raise ValueError("'final_avg_pseudotime' was not calculated. Check scoring logic.")

    # Assign to adata
    embedding_adata.obs['final_avg_pseudotime'] = pseudotime_df['final_avg_pseudotime']
    gat_epi.obs['final_avg_pseudotime'] = embedding_adata.obs['final_avg_pseudotime']

    # Plot histogram
    gat_epi.obs['final_avg_pseudotime'].hist()

    return

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
