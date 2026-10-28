# PlastiNet

## Authors

**Package Authors**
Izabella Zamora and Lucy Zhang

The development of PlastiNet was part of the study published as: **PlastiNet: Understanding the Epithelial-Mesenchymal Transition Through Graph Attention in Spatial Transcriptomics**
Izabella Zamora, Lynn Bi, Milan Parikh, Abigail Collins, Samuel Wright, Bidish K. Patel, Martin Hemberg, William L. Hwang, Aziz Al’Khafaji, David Ting, Arnav Mehta, Nir Hacohen

## Overview

PlastiNet is a graph attention-based network (GAT) designed to analyze spatial transcriptomics data and uncover the mechanisms of cellular plasticity in the tumor microenvironment. By leveraging spatial and transcriptional information, PlastiNet generates spatially aware embeddings, allowing researchers to identify cellular neighborhoods, construct pseudotime trajectories, and reveal key cell-cell interactions that drive state transitions such as the epithelial-mesenchymal transition (EMT).

PlastiNet has been validated on multiple datasets, including colon and brain and applied to pancreatic ductal adenocarcinoma (PDAC) samples, demonstrating its ability to:

- Integrate spatial and transcriptional data to overcome gene panel limitations.
- Identify plasticity spectra and key signaling axes
- Propose actionable hypotheses and therapeutic targets.

![PlastiNetv2 Overview Schematic](https://github.com/user-attachments/assets/b569fcd8-e968-418f-a30e-42d0278fb4ce)

## Key Features

### Input

An `AnnData` object with:

- expression in `.X` (dense or sparse), one column per gene in the panel;
- cell coordinates in `.obsm["spatial"]`;
- a column in `.obs` naming the tissue section each cell comes from (`sample_key`).

### Spatial graph

Each section gets its own graph, so no edges cross sections. A cell is linked to every other cell whose centre lies within `radius` of it, in the same units as the coordinates. The graph is undirected and has no self-loops. `plot_subsample_radius` draws the radius on a sample of cells to help pick a value.

### Encoder

Two stacked self/neighbor attention layers work in gene space: their input and output both have one value per gene, so every attention weight belongs to a specific gene. For a cell *i* with neighbors *j*, each layer:

1. applies `LayerNorm` to the cell and its neighbors;
2. scores the cell (self attention) and each neighbor (neighbor attention) with separate linear layers, giving one score per gene;
3. subtracts a distance penalty `alpha * d_ij / radius` from each neighbor's score. This is the same as multiplying the neighbor's unnormalized weight by `exp(-alpha * d_ij / radius)`, so nearer neighbors are favored;
4. applies a softmax per gene across the cell and all its neighbors together;
5. outputs `beta * (w_self ⊙ x_i) + (1 - beta) * Σ_j w_ij ⊙ x_j`, followed by LeakyReLU.

Layer 1 uses `beta_1` and starts from the cell's own expression. Layer 2 uses `beta_2` and takes layer 1's output as the self term. Both layers attend over the neighbors' input expression. A final linear layer reduces the genes to a `z_dim`-dimensional embedding (default 50).

### Loss function

`total = DGI + spatial_reg * spatial + l1_reg * L1`

- **Deep Graph Infomax (DGI):** contrasts each cell's embedding with a summary of the whole section (`sigmoid` of the mean embedding). The corrupted graph keeps the real edges but zeroes each input feature at random: a feature is kept with probability `mask_n`, so the default `mask_n=0.7` masks about 30% of features.
- **Spatial regularization:** draws `spatial_percent × n_cells` random pairs of cells from a section and scales both their embedding distances and their spatial distances by the largest value in the draw. The loss is `mean((1 - d_embedding) * d_spatial)`, which penalizes cells that are far apart in the tissue but close in the embedding.
- **L1 regularization:** the L1 norm of the self and neighbor attention scoring layers' parameters. This pushes the attention toward fewer genes.

Training uses Adam with `weight_decay` and a `StepLR` schedule that multiplies the learning rate by `gamma` every `step_size` epochs. Each step processes one tissue section. Training stops early when the total epoch loss hasn't improved for `patience` epochs, and the best weights are restored.

### Outputs

`PlastiNet.run_gat()` returns an `AnnData` whose rows are in the same order as the input:

| Field | Contents |
| --- | --- |
| `.X` | embedding, `n_cells × z_dim` |
| `.obs` | copy of the input `.obs` |
| `.obsm["self_attention_weights_layer1"]`, `["..._layer2"]` | self attention per cell and gene, `n_cells × n_genes` |
| `.uns["neighbor_attention"]` | edge list: `target` and `source` are row positions (the attention `target` pays to neighbor `source`); `layer1` and `layer2` are `n_edges × n_genes` weights |
| `.uns["reduction_layes"]` | `weights` and `biases` of the final linear layer |

### Downstream analysis

- **Neighborhoods** (`analysis.neighborhood_analysis`): Leiden clustering of the embedding, top genes per cluster, and how often each neighborhood lies within a radius of each other neighborhood.
- **Pseudotime** (`analysis.pseudotime.construct_differentiation_path`): within one cell type, score a starting gene set with `sc.tl.score_genes` and run diffusion pseudotime (DPT) on the embedding's kNN graph from each of the top `N` cells. Then run it again, inverted, from the `N` cells with the highest end-state score; this uses the lowest starting scores when no end gene set is given. The runs are averaged and scaled to [0, 1] in `obs["final_avg_pseudotime"]`. `plot_pseudotime_heatmap` bins the result, and `plot_gam_curves` fits GAMs (`pygam`) of gene expression along it.
- **Pseudotime states** (`analysis.spatial_colocalization`): a Gaussian mixture splits pseudotime into discrete states, followed by colocalization of those states with cell types and neighborhoods, and Moran's I spatial autocorrelation.
- **Attention** (`analysis.attention_analysis`):
  - `analyze_self_attention_layer` gives each cell type's self attention per gene.
  - `analyze_attention_flow` multiplies layer 1 by layer 2 for each gene and averages by cell type.
  - `analyze_attention_flow_with_permutation` adds a two-sided label-permutation p-value.
  - `analyze_stromal_to_epi_attention` follows the neighbor attention one cell type pays another along pseudotime.

## Installation

Install from source to get the latest fixes:

```bash
git clone https://github.com/izabellaleahz/plastinet.git
cd plastinet
pip install -e .
```

`pip install plastinet` installs the most recent PyPI release, which may lag behind `main`.

PyTorch and PyTorch Geometric are installed from PyPI by default. For a specific CUDA version, install them first, following the [PyTorch](https://pytorch.org/get-started/locally/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) instructions.

### Conda environment

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download).
2. Create the environment: `conda env create --file=environment.yaml`
3. Keep the environment isolated from packages installed with `pip install --user`:
   `conda env config vars set -n plastiNet PYTHONNOUSERSITE=1`
4. `conda activate plastiNet`
5. Install PlastiNet into it: `pip install -e .`
6. [Optional] To use it in Jupyter, register the kernel: `python -m ipykernel install --user --name=plastiNet --env PYTHONNOUSERSITE 1`

**Troubleshooting: packages in `~/.local`**

The environment uses Python 3.10. If your system Python is also 3.10, both read packages from `~/.local/lib/python3.10/site-packages`. Without step 3, the environment can import those packages instead of its own. Worse, `pip` inside it can uninstall them, which breaks your system Python. If that happened before you set the variable, reinstall what was removed with your system Python, e.g. `python3 -m pip install --user torch_geometric==<version>`.

## Usage

```python
from plastinet.models.plastinet_model import PlastiNet

model = PlastiNet(adata, sample_key="sample_source", radius=200)
embedding_adata = model.run_gat()
```

For the full workflow, from training through neighborhoods, pseudotime and attention analysis, open [tutorial_on_ex_data.ipynb](https://github.com/izabellaleahz/plastinet/blob/main/notebooks/tutorial_on_ex_data.ipynb) with the `plastiNet` kernel selected. It loads the bundled `notebooks/testing_data.h5ad` and finds the repo from its own location, so no path needs editing unless you move the notebook.

## Citation

If you use PlastiNet in your research, please cite:

PlastiNet: Understanding the Epithelial-Mesenchymal Transition Through Graph Attention in Spatial Transcriptomics
Izabella Zamora, et al. (2024).

## Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue for bug reports and feature requests.

## License

This project is licensed under the [MIT License](https://opensource.org/license/mit).

## Contact

For questions or support, please contact zamora@broadinstitute.org
