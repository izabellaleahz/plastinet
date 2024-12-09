# plastinet/analysis/pseudotime.py
import scanpy as sc
import pandas as pd
import numpy as np

def construct_differentiation_path(embedding_adata, adata, cell_type_obs, cell_type, starting_point_gene_list, end_point_gene_list=None, N=5): 
    gat_epi = embedding_adata[embedding_adata.obs[cell_type_obs] == cell_type]
    exp_epi = adata[adata.obs[cell_type_obs] == cell_type]
    
    pseudotime_df = pd.DataFrame(index=gat_epi.obs_names)

    sc.tl.score_genes(exp_epi, starting_point_gene_list, score_name='starting_score')
    gat_epi.obs["starting_score"] = exp_epi.obs["starting_score"]
    top_cells_indices = gat_epi.obs["starting_score"].nlargest(N).index

    sc.pp.neighbors(gat_epi, use_rep='X')
    sub_adata = gat_epi.copy()

    for idx, top_cell in enumerate(top_cells_indices, start=1):
        sub_adata.uns['iroot'] = np.flatnonzero(sub_adata.obs_names == top_cell)[0]
        sc.tl.dpt(sub_adata, n_branchings=0)

        pseudotime_key = f'dpt_pseudotime_global_{idx}'
        pseudotime_df[pseudotime_key] = sub_adata.obs['dpt_pseudotime'].reindex(pseudotime_df.index)

    classical_keys = [f'dpt_pseudotime_global_{i}' for i in range(1, N+1)]
    pseudotime_df['avg_start_pseudotime'] = pseudotime_df[classical_keys].mean(axis=1, skipna=True)

 
    if end_point_gene_list is None:
        exp_epi.obs['neg_starting_score'] = -exp_epi.obs['starting_score']
        gat_epi.obs['neg_starting_score'] = exp_epi.obs['neg_starting_score']
        top_negative_indices = gat_epi.obs["neg_starting_score"].nlargest(N).index

        for idx, top_cell in enumerate(top_negative_indices, start=1):
            sub_adata.uns['iroot'] = np.flatnonzero(sub_adata.obs_names == top_cell)[0]
            sc.tl.dpt(sub_adata, n_branchings=0)

            max_pseudotime = sub_adata.obs['dpt_pseudotime'].max()
            inverted_pseudotime_key = f'inverted_dpt_pseudotime_global_{idx}'
            pseudotime_df[inverted_pseudotime_key] = max_pseudotime - sub_adata.obs['dpt_pseudotime'].reindex(pseudotime_df.index)

        inverted_negative_keys = [f'inverted_dpt_pseudotime_global_{i}' for i in range(1, N+1)]
        pseudotime_df['avg_inverted_neg_pseudotime'] = pseudotime_df[inverted_negative_keys].mean(axis=1, skipna=True)

        pseudotime_df['final_avg_pseudotime'] = pseudotime_df[['avg_start_pseudotime', 'avg_inverted_neg_pseudotime']].mean(axis=1, skipna=True)

    else:

        sc.tl.score_genes(exp_epi, end_point_gene_list, score_name='ending_score')
        gat_epi.obs["ending_score"] = exp_epi.obs["ending_score"]
        top_basal_indices = gat_epi.obs["ending_score"].nlargest(N).index

        for idx, top_cell in enumerate(top_basal_indices, start=1):
            sub_adata.uns['iroot'] = np.flatnonzero(sub_adata.obs_names == top_cell)[0]
            sc.tl.dpt(sub_adata, n_branchings=0)

            max_pseudotime = sub_adata.obs['dpt_pseudotime'].max()
            inverted_pseudotime_key = f'inverted_dpt_pseudotime_global_{idx}'
            pseudotime_df[inverted_pseudotime_key] = max_pseudotime - sub_adata.obs['dpt_pseudotime'].reindex(pseudotime_df.index)

        inverted_basal_keys = [f'inverted_dpt_pseudotime_global_{i}' for i in range(1, N+1)]
        pseudotime_df['avg_inverted_end_pseudotime'] = pseudotime_df[inverted_basal_keys].mean(axis=1, skipna=True)

        
    pseudotime_df['final_avg_pseudotime'] = (
    pseudotime_df['final_avg_pseudotime'] - pseudotime_df['final_avg_pseudotime'].min()
) / (
    pseudotime_df['final_avg_pseudotime'].max() - pseudotime_df['final_avg_pseudotime'].min()
)

    embedding_adata.obs['final_avg_pseudotime'] = pseudotime_df['final_avg_pseudotime']
    gat_epi.obs["final_avg_pseudotime"] = embedding_adata.obs['final_avg_pseudotime']
    gat_epi.obs["final_avg_pseudotime"].hist()
    
    return 

