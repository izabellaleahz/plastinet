"""Regression tests for bugs fixed in the graph construction, encoder, training loop and attention analysis."""
import anndata
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.loader import DataLoader

from plastinet.analysis import attention_analysis
from plastinet.data.data_loader import create_data_objects
from plastinet.data.graph_utils import graph_alpha, graph_to_sparse_matrix, sparse_mx_to_torch_edge_list
from plastinet.models.attention import GraphAttentionEncoder, SelfNeighborAttention
from plastinet.models.plastinet_model import PlastiNet

N_GENES = 12
RADIUS = 100.0


def edge_index_for(coords, radius=RADIUS):
    return sparse_mx_to_torch_edge_list(graph_to_sparse_matrix(graph_alpha(coords, radius))).long()


def make_encoder(seed=0):
    torch.manual_seed(seed)
    return GraphAttentionEncoder(
        N_GENES, 6, RADIUS, dropout_rate=0.0, beta_1=0.5, beta_2=0.5, alpha=3.0, attention_threshold=0.01
    ).eval()


def random_tissue(n_cells, seed=0, extent=600.0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, extent, (n_cells, 2))
    x = rng.normal(size=(n_cells, N_GENES)).astype(np.float32)
    return coords, x


def make_adata(n_per_sample=60, samples=("s1", "s2", "s3"), seed=0):
    """Samples are interleaved row-wise so that sample order differs from adata order."""
    rng = np.random.default_rng(seed)
    n = n_per_sample * len(samples)
    obs = pd.DataFrame(
        {
            "sample": np.tile(samples, n_per_sample),
            "cell_type": rng.choice(["epi", "stroma", "immune"], size=n),
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    adata = anndata.AnnData(rng.normal(size=(n, N_GENES)).astype(np.float32), obs=obs)
    adata.var_names = [f"g{i}" for i in range(N_GENES)]
    adata.obsm["spatial"] = rng.uniform(0, 500, (n, 2))
    return adata


def make_plastinet(adata, **kwargs):
    params = dict(sample_key="sample", radius=RADIUS, z_dim=6, epochs=2, patience=10, random_seed=0)
    params.update(kwargs)
    return PlastiNet(adata, **params)


# Bug 1: the spatial graph must be symmetric and independent of cell order

def test_edge_index_is_symmetric_and_matches_graph_degree():
    coords, _ = random_tissue(300)
    G = graph_alpha(coords, RADIUS)
    row, col = edge_index_for(coords)

    assert row.numel() == 2 * G.number_of_edges()
    edges = set(zip(row.tolist(), col.tolist()))
    assert all((j, i) in edges for i, j in edges)
    assert np.array_equal(torch.bincount(row, minlength=300).numpy(), [G.degree(i) for i in range(300)])


def test_graph_without_edges():
    G = nx.empty_graph(3)
    edge_index = sparse_mx_to_torch_edge_list(graph_to_sparse_matrix(G))
    assert tuple(edge_index.shape) == (2, 0)


def test_embedding_is_equivariant_to_cell_order():
    coords, x = random_tissue(200)
    perm = np.random.default_rng(1).permutation(200)
    encoder = make_encoder()

    with torch.no_grad():
        z = encoder(torch.tensor(x), edge_index_for(coords), torch.tensor(coords, dtype=torch.float))
        z_perm = encoder(
            torch.tensor(x[perm]), edge_index_for(coords[perm]), torch.tensor(coords[perm], dtype=torch.float)
        )

    torch.testing.assert_close(z_perm, z[perm], rtol=1e-4, atol=1e-5)


# Bug 2: a cell's own expression must drive its own embedding

def test_own_expression_changes_own_embedding_and_only_neighbors():
    coords, x = random_tissue(200)
    edge_index = edge_index_for(coords)
    pos = torch.tensor(coords, dtype=torch.float)
    encoder = make_encoder()

    cell = int(torch.bincount(edge_index[0], minlength=200).argmax())
    x_changed = x.copy()
    x_changed[cell] = np.random.default_rng(2).normal(scale=5.0, size=N_GENES)

    with torch.no_grad():
        z = encoder(torch.tensor(x), edge_index, pos)
        z_changed = encoder(torch.tensor(x_changed), edge_index, pos)

    changed = set(((z_changed - z).abs().max(dim=1).values > 1e-6).nonzero().flatten().tolist())
    neighbors = set(nx.Graph(graph_alpha(coords, RADIUS)).neighbors(cell))
    assert cell in changed
    assert changed == neighbors | {cell}


# Bug 5: nearer neighbors get more attention even when scores are negative

def test_distance_penalty_favors_nearer_neighbors_for_negative_scores():
    layer = SelfNeighborAttention(N_GENES, RADIUS, beta=0.5, alpha=3.0)
    with torch.no_grad():
        layer.neighbor_attn_layer.weight.zero_()
        layer.neighbor_attn_layer.bias.fill_(-5.0)

    target = torch.randn(1, N_GENES)
    neighbors = torch.randn(1, 1, N_GENES).repeat(1, 2, 1)
    distances = torch.tensor([[10.0, 90.0]])

    with torch.no_grad():
        _, _, neighbor_weights = layer(target, neighbors, distances)

    assert torch.all(neighbor_weights[0, 0] > neighbor_weights[0, 1])


# Bug 6: padded neighbor slots get zero attention and don't change the output

def test_padding_does_not_affect_embedding():
    # Component A: a pair of cells. Component B: a dense cluster far away, which forces padding for A.
    coords_a = np.array([[0.0, 0.0], [30.0, 0.0]])
    coords_b = np.array([[5000.0 + dx, 5000.0 + dy] for dx in (0, 20, 40) for dy in (0, 20, 40)])
    coords = np.vstack([coords_a, coords_b])
    x = np.random.default_rng(3).normal(size=(len(coords), N_GENES)).astype(np.float32)
    encoder = make_encoder()

    with torch.no_grad():
        z_all = encoder(torch.tensor(x), edge_index_for(coords), torch.tensor(coords, dtype=torch.float))
        padded_weights = encoder.neighbor_attention_weights1[:2, 1:]
        z_alone = encoder(torch.tensor(x[:2]), edge_index_for(coords_a), torch.tensor(coords_a, dtype=torch.float))

    assert torch.all(padded_weights == 0)
    torch.testing.assert_close(z_all[:2], z_alone, rtol=1e-5, atol=1e-6)


# Bug 3: training restores the best epoch's parameters, not the last epoch's

def test_train_restores_best_epoch_parameters(monkeypatch):
    adata = make_adata(samples=("s1",))
    plastinet = make_plastinet(adata, epochs=3, spatial_reg=1.0)
    dataloader = DataLoader(create_data_objects(adata, "sample", RADIUS), batch_size=1, shuffle=False)

    original_spatial_loss = plastinet.compute_spatial_loss
    snapshots = []

    def spatial_loss_with_bad_later_epochs(z, coords, subset_percent):
        # One batch per epoch: on call k, the parameters are the ones left by epoch k - 1
        snapshots.append({k: v.detach().clone() for k, v in plastinet.model.state_dict().items()})
        penalty = 0.0 if len(snapshots) == 1 else 1000.0
        return original_spatial_loss(z, coords, subset_percent) + penalty

    monkeypatch.setattr(plastinet, "compute_spatial_loss", spatial_loss_with_bad_later_epochs)
    plastinet.train(dataloader)

    after_epoch_1 = snapshots[1]
    final_state = plastinet.model.state_dict()
    for key, value in after_epoch_1.items():
        torch.testing.assert_close(final_state[key], value)


# Bug 4: embeddings and attention line up with adata rows when there are several samples

def test_embedding_rows_match_adata_with_multiple_samples():
    adata = make_adata()
    plastinet = make_plastinet(adata, epochs=1)
    embedding_adata = plastinet.run_gat()

    assert list(embedding_adata.obs_names) == list(adata.obs_names)

    encoder = plastinet.model.encoder
    for sample in ("s1", "s2", "s3"):
        mask = (adata.obs["sample"] == sample).to_numpy()
        coords = adata.obsm["spatial"][mask]
        with torch.no_grad():
            z = encoder(
                torch.tensor(adata.X[mask]).to(plastinet.device),
                edge_index_for(coords).to(plastinet.device),
                torch.tensor(coords, dtype=torch.float).to(plastinet.device),
            )
        np.testing.assert_allclose(embedding_adata.X[mask], z.cpu().numpy(), rtol=1e-4, atol=1e-5)

    # Neighbor attention edges connect true spatial neighbors from the same sample, in global row positions
    neighbor_attention = embedding_adata.uns["neighbor_attention"]
    targets, sources = neighbor_attention["target"], neighbor_attention["source"]
    expected_edges = set()
    for sample in ("s1", "s2", "s3"):
        rows = np.flatnonzero((adata.obs["sample"] == sample).to_numpy())
        G = graph_alpha(adata.obsm["spatial"][rows], RADIUS)
        expected_edges |= {(rows[i], rows[j]) for i, j in G.edges()} | {(rows[j], rows[i]) for i, j in G.edges()}
    assert set(zip(targets.tolist(), sources.tolist())) == expected_edges
    assert neighbor_attention["layer1"].shape == (len(targets), N_GENES)

    # Each cell's neighbor attention plus its self attention sums to 1 per gene
    totals = embedding_adata.obsm["self_attention_weights_layer1"].copy()
    np.add.at(totals, targets, neighbor_attention["layer1"])
    np.testing.assert_allclose(totals, 1.0, rtol=1e-4)


# Bug 7: stromal -> epithelial neighbor attention uses the right cells, neighbors and genes

def test_stromal_to_epi_attention_averages_source_neighbors():
    obs = pd.DataFrame(
        {"cell_type": ["epi", "epi", "stroma", "stroma", "immune"], "pt": [0.9, 0.1, np.nan, np.nan, np.nan]},
        index=[f"c{i}" for i in range(5)],
    )
    adata = anndata.AnnData(np.zeros((5, 3), dtype=np.float32), obs=obs)
    adata.var_names = ["A", "B", "C"]
    embedding_adata = anndata.AnnData(np.zeros((5, 2), dtype=np.float32), obs=obs.copy())
    # Rows: c0<-c2, c0<-c3, c0<-c4 (immune, ignored), c1<-c3
    embedding_adata.uns["neighbor_attention"] = {
        "target": np.array([0, 0, 0, 1]),
        "source": np.array([2, 3, 4, 3]),
        "layer1": np.array([[0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [0.9, 0.9, 0.9], [0.6, 0.7, 0.8]]),
        "layer2": np.zeros((4, 3)),
    }

    result = attention_analysis.analyze_stromal_to_epi_attention(
        embedding_adata, adata, pseudotime_key="pt", cell_type_col="cell_type",
        source_type="stroma", target_type="epi", gene_list=["C", "A"], normalize=False,
    )

    assert list(result.columns) == ["C", "A"]
    assert list(result.index) == [0.1, 0.9]
    np.testing.assert_allclose(result.to_numpy(), [[0.8, 0.6], [0.4, 0.2]])


# Bug 8: the permutation test can detect a real difference

def test_attention_flow_permutation_detects_enriched_cell_type():
    rng = np.random.default_rng(4)
    n = 200
    cell_type = np.array(["high"] * 50 + ["rest"] * 150)
    flow = rng.uniform(0.4, 0.6, size=(n, 2))
    flow[:50, 0] += 0.3  # gene g0 enriched in "high"

    obs = pd.DataFrame({"cell_type": cell_type}, index=[f"c{i}" for i in range(n)])
    embedding_adata = anndata.AnnData(np.zeros((n, 2), dtype=np.float32), obs=obs)
    embedding_adata.obsm["self_attention_weights_layer1"] = np.sqrt(flow)
    embedding_adata.obsm["self_attention_weights_layer2"] = np.sqrt(flow)
    adata = anndata.AnnData(np.zeros((n, 2), dtype=np.float32), obs=obs.copy())
    adata.var_names = ["g0", "g1"]

    pvals = attention_analysis.analyze_attention_flow_with_permutation(
        embedding_adata, adata, cell_type_col="cell_type", gene_list=["g0", "g1"], n_permutations=200
    )

    assert pvals.loc["g0", "high"] < 0.01
    assert pvals.loc["g1", "high"] > 0.05


# NaN cell types must not shift which rows are averaged

def test_attention_flow_with_nan_cell_types_uses_aligned_rows():
    obs = pd.DataFrame({"cell_type": [np.nan, "a", "a", "b"]}, index=[f"c{i}" for i in range(4)])
    weights = np.array([[100.0], [1.0], [3.0], [10.0]])
    embedding_adata = anndata.AnnData(np.zeros((4, 1), dtype=np.float32), obs=obs)
    embedding_adata.obsm["self_attention_weights_layer1"] = weights
    embedding_adata.obsm["self_attention_weights_layer2"] = np.ones_like(weights)
    adata = anndata.AnnData(np.zeros((4, 1), dtype=np.float32), obs=obs.copy())
    adata.var_names = ["g0"]

    result = attention_analysis.analyze_attention_flow(
        embedding_adata, adata, cell_type_col="cell_type", gene_list=["g0"], normalize=False
    )

    assert result.loc["g0", "a"] == pytest.approx(2.0)
    assert result.loc["g0", "b"] == pytest.approx(10.0)


def test_attention_flow_zscore_returns_dataframe():
    obs = pd.DataFrame({"cell_type": ["a", "a", "b", "b"]}, index=[f"c{i}" for i in range(4)])
    weights = np.array([[1.0, 4.0], [3.0, 2.0], [10.0, 1.0], [12.0, 3.0]])
    embedding_adata = anndata.AnnData(np.zeros((4, 1), dtype=np.float32), obs=obs)
    embedding_adata.obsm["self_attention_weights_layer1"] = weights
    embedding_adata.obsm["self_attention_weights_layer2"] = np.ones_like(weights)
    adata = anndata.AnnData(np.zeros((4, 2), dtype=np.float32), obs=obs.copy())
    adata.var_names = ["g0", "g1"]

    result = attention_analysis.analyze_attention_flow(
        embedding_adata, adata, cell_type_col="cell_type", gene_list=["g0", "g1"], normalize=True
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result.loc["g0"].to_numpy(), [-1.0, 1.0])
    # analyze_self_attention_layer uses the same normalization path
    attention_analysis.analyze_self_attention_layer(
        embedding_adata, adata, cell_type_col="cell_type", gene_list=["g0", "g1"], normalize=True
    )


def test_gmm_labels_are_ordered_by_mean_pseudotime():
    from plastinet.analysis import spatial_colocalization

    rng = np.random.default_rng(5)
    pt = np.concatenate([rng.normal(0.1, 0.03, 300), rng.normal(0.5, 0.03, 300), rng.normal(0.9, 0.03, 300)])
    adata = anndata.AnnData(np.zeros((len(pt), 1), dtype=np.float32),
                            obs=pd.DataFrame({"pt": rng.permutation(pt)}, index=[f"c{i}" for i in range(len(pt))]))
    spatial_colocalization.gmm_discretize_pseudotime(adata, "pt", n_components=3, plot=False)
    means = adata.obs.groupby("GMM_labels").pt.mean()
    assert list(means.index) == [1, 2, 3]
    assert means.is_monotonic_increasing
