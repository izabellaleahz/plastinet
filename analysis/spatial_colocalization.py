import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree


def gmm_discretize_pseudotime(adata, pseudotime_key, n_components=3, output_label_key="GMM_labels", plot=True):

    pseudotime = adata.obs[pseudotime_key].values
    valid_indices = ~np.isnan(pseudotime)
    valid_pseudotime = pseudotime[valid_indices]

    if valid_pseudotime.size == 0:
        raise ValueError("No valid pseudotime values to fit GMM.")

    pseudotime_reshaped = valid_pseudotime.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(pseudotime_reshaped)

    labels = gmm.predict(pseudotime_reshaped)
    full_labels = np.full(pseudotime.shape, np.nan)
    full_labels[valid_indices] = labels
    adata.obs[output_label_key] = [int(x) + 1 if not np.isnan(x) else np.nan for x in full_labels]

    if plot:
        plot_gmm_pseudotime(valid_pseudotime, gmm, n_components)

def plot_gmm_pseudotime(valid_pseudotime, gmm, n_components):
    means = gmm.means_.ravel()
    covariances = gmm.covariances_.ravel()
    weights = gmm.weights_.ravel()

    plt.figure(figsize=(10, 6))
    plt.hist(valid_pseudotime, bins=300, density=True, alpha=0.5, color="skyblue", label="Histogram")
    x = np.linspace(min(valid_pseudotime), max(valid_pseudotime), 1000)

    for i in range(n_components):
        curve = weights[i] * (
            1 / (np.sqrt(2 * np.pi * covariances[i])) *
            np.exp(-0.5 * ((x - means[i]) / np.sqrt(covariances[i])) ** 2)
        )
        plt.plot(x, curve, label=f"Component {i + 1}", color=f"C{i + 2}")

    plt.xlabel("Pseudotime")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Pseudotime Distribution with Gaussian Mixture Model")
    plt.grid()
    plt.show()


def calculate_colocalization(
    adata, x_key="X", y_key="Y", cell_type_key="cellTypes", subtype_key="subtypes", cancer_label = "epi", radius=0.05
):

    spatial_locs = np.stack((adata.obs[x_key], adata.obs[y_key]), axis=1)
    cell_types = adata.obs[cell_type_key].to_numpy()
    hmm_states = adata.obs[subtype_key].to_numpy()

    combined_labels = np.array([
        f"{ct}: {hmm}" if ct == cancer_label and pd.notna(hmm) else ct
        for ct, hmm in zip(cell_types, hmm_states)
    ])

    kd_tree = cKDTree(spatial_locs)
    unique_labels = np.unique(combined_labels)
    neighbor_counts = {label: {} for label in unique_labels}

    for cell_index, (loc, label) in enumerate(zip(spatial_locs, combined_labels)):
        indices = kd_tree.query_ball_point(loc, radius)
        indices = [i for i in indices if i != cell_index]
        neighbor_labels = combined_labels[indices]

        for neighbor_label in neighbor_labels:
            neighbor_counts[label][neighbor_label] = neighbor_counts[label].get(neighbor_label, 0) + 1

    return {label: pd.Series(counts).sort_index() for label, counts in neighbor_counts.items()}

def plot_stacked_colocalization(output, normalize=True):

    colocalization_df = pd.DataFrame(output).fillna(0)

    if normalize:
        colocalization_df = colocalization_df.div(colocalization_df.sum(axis=1), axis=0)

    colocalization_df.plot(
        kind="bar", stacked=True, figsize=(12, 8), colormap="tab20", edgecolor="black"
    )

    plt.title("Spatial Colocalization of Cell Types")
    plt.xlabel("Cell Types")
    plt.ylabel("Proportion" if normalize else "Count")
    plt.legend(title="Neighbor Cell Types", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def calculate_gmm_colocalization(
    adata, x_key="X", y_key="Y", leiden_key="leiden", gmm_key="GMM_labels", radius=200
):
    spatial_locs = np.stack((adata.obs[x_key], adata.obs[y_key]), axis=1)
    leiden_labels = adata.obs[leiden_key].astype(str).to_numpy()
    gmm_labels = adata.obs[gmm_key].astype(str).to_numpy()
    kd_tree = cKDTree(spatial_locs)

    unique_gmm_labels = np.unique(gmm_labels)
    unique_neighborhoods = np.unique(leiden_labels)

    neighbor_counts = {gmm_label: {nbr: 0 for nbr in unique_neighborhoods} for gmm_label in unique_gmm_labels}

    for cell_index, (loc, gmm_label) in enumerate(zip(spatial_locs, gmm_labels)):
        if pd.isna(gmm_label):  
            continue
        indices = kd_tree.query_ball_point(loc, radius)
        indices = [i for i in indices if i != cell_index]  
        neighbor_labels = leiden_labels[indices]

        for neighbor_label in neighbor_labels:
            neighbor_counts[gmm_label][neighbor_label] += 1

    neighbor_counts_df = pd.DataFrame(neighbor_counts).fillna(0)

    normalized_counts_df = neighbor_counts_df.div(neighbor_counts_df.sum(axis=0), axis=1)

    return normalized_counts_df
    
def plot_gmm_colocalization(colocalization_df):
    colocalization_df = colocalization_df.loc[
        ~colocalization_df.index.isna(), 
        ~colocalization_df.columns.isna() & (colocalization_df.columns != "nan")
    ]
    colocalization_df = colocalization_df[colocalization_df.sum(axis=1) > 0]

    colocalization_df.T.plot(
        kind="bar", stacked=True, figsize=(12, 8), colormap="tab20", edgecolor="black"
    )

    plt.title("GMM Cluster Colocalization with Neighborhoods")
    plt.xlabel("GMM Clusters")
    plt.ylabel("Proportion")
    plt.legend(title="Neighbor Neighborhoods", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import pandas as pd

def compute_spatial_autocorrelation(adata, x_key="X", y_key="Y", gmm_key="GMM_labels"):
    
    spatial_coords = adata.obs[[x_key, y_key]].to_numpy()
    gmm_states = adata.obs[gmm_key]

    valid_mask = ~gmm_states.isna()
    spatial_coords = spatial_coords[valid_mask]
    gmm_states = gmm_states[valid_mask]

    distance_matrix = euclidean_distances(spatial_coords)

    weights_matrix = 1 / (distance_matrix + np.eye(distance_matrix.shape[0]))  
    np.fill_diagonal(weights_matrix, 0) 

    moran_i_results = {}
    unique_gmm_states = gmm_states.unique()

    for state in unique_gmm_states:
        gmm_binary = (gmm_states == state).astype(float).to_numpy()

        z = gmm_binary - gmm_binary.mean()
        numerator = np.sum(weights_matrix * np.outer(z, z))
        denominator = np.sum(z**2)
        weights_sum = np.sum(weights_matrix)

        moran_i = (len(z) / weights_sum) * (numerator / denominator)
        moran_i_results[state] = moran_i

    return pd.Series(moran_i_results)

