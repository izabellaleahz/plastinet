#  PlastiNet

## Authors

**Package Authors**
Izabella Zamora and Lucy Zhang

**Paper Authors**

The development of PlastiNet was part of the study published as:
PlastiNet: Understanding the Epithelial-Mesenchymal Transition Through Graph Attention in Spatial Transcriptomics
Izabella Zamora, Lynn Bi, Milan Parikh, Abigail Collins, Samuel Wright, Bidish K. Patel, Martin Hemberg, William L. Hwang, Aziz Al’Khafaji, David Ting, Arnav Mehta, Nir Hacohen

## Overview

PlastiNet is a graph attention-based network (GAT) designed to analyze spatial transcriptomics data and uncover the mechanisms of cellular plasticity in the tumor microenvironment. By leveraging spatial and transcriptional information, PlastiNet generates spatially aware embeddings, allowing researchers to identify cellular neighborhoods, construct pseudotime trajectories, and reveal key cell-cell interactions that drive state transitions such as the epithelial-mesenchymal transition (EMT).

PlastiNet has been validated on multiple datasets, including colon and brain and applied to pancreatic ductal adenocarcinoma (PDAC) samples, demonstrating its ability to:

- Integrate spatial and transcriptional data to overcome gene panel limitations.
- Identify plasticity spectra and key signaling axes
- Propose actionable hypotheses and therapeutic targets.

## Key Features

**Spatial Graph Construction:** Nodes represent cells, and edges connect cells within a predefined spatial radius, capturing local neighborhood interactions.
Custom GAT Architecture:
- Incorporates self-attention (intrinsic features) and neighbor-attention (extrinsic influences).
- Applies distance-weighted attention to prioritize nearby neighbors.
- Produces spatially aware embeddings via learned reduction layers.

**Loss Function:**
- Spatial Regularization: Preserves spatial relationships in the embedding space.
- L1 Regularization: Promotes sparsity to enhance interpretability and feature selection.
- Deep Graph Infomax (DGI): Ensures high-quality node embeddings by contrasting real and corrupted graph data.

**Downstream Analysis:**
- Clustering of cells into spatial neighborhoods.
- Pseudotime trajectory construction to track plasticity transitions.
- Attention weight analysis to identify key cell-cell interactions.

## Requirements

1. Download [Anaconda](https://anaconda.org/)
2. Create an environment from deps: `conda env create --name plastiNet --file=environment.yaml`
3. `conda activate plastiNet`
4. [Optional]: To use in Jupyter Notebook, add environment to ipykernel instance: `python -m ipykernel install --user --name=plastiNet`

## Usage

Open the [tutorial_on_ex_data.ipynb](https://github.com/izabellaleahz/plastinet/blob/main/notebooks/tutorial_on_ex_data.ipynb) and run.

## Publishing

TODO: Lucy to add to this section

## Citation
If you use PlastiNet in your research, please cite:

PlastiNet: Understanding the Epithelial-Mesenchymal Transition Through Graph Attention in Spatial Transcriptomics
Izabella Zamora, et al. (2024). 

## Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue for bug reports and feature requests.

## License

This project is licensed under the [MIT License](https://opensource.org/license/mit).

## Contact

For questions or support, please contact

zamora@broadinstitute.org 
