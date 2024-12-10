import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class SelfNeighborAttention(nn.Module):
    def __init__(self, gene_dim, reduced_dim, radius, beta=0.5, alpha=2, attention_threshold=0.01):
        super(SelfNeighborAttention, self).__init__()
        self.gene_dim = gene_dim
        self.reduced_dim = reduced_dim
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
        self.alpha = alpha
        self.radius = radius
        self.attention_threshold = attention_threshold

        self.layer_norm = nn.LayerNorm(gene_dim)
        self.self_attn_layer = nn.Linear(gene_dim, gene_dim)
        self.neighbor_attn_layer = nn.Linear(gene_dim, gene_dim)
        self.reduction_attn_layer = nn.Linear(gene_dim, reduced_dim)

        self._initialize_weights()

    def forward(self, target_cell, neighbors, distances):
        # Normalize inputs
        target_cell = self.layer_norm(target_cell)
        neighbors = self.layer_norm(neighbors)

        # Compute attention scores
        self_attn_scores = self.self_attn_layer(target_cell)
        neighbor_attn_scores = self.neighbor_attn_layer(neighbors)

        # Apply distance weighting
        normalized_distances = distances / (self.radius + 1e-8)
        distance_weights = torch.exp(-self.alpha * normalized_distances)
        weighted_neighbor_scores = neighbor_attn_scores * distance_weights.unsqueeze(-1)

        # Combine scores and compute weights
        combined_scores = torch.cat([self_attn_scores.unsqueeze(1), weighted_neighbor_scores], dim=1)
        combined_weights = F.softmax(combined_scores, dim=1)

        self_attn_weights = combined_weights[:, 0, :]
        neighbor_attn_weights = combined_weights[:, 1:, :]

        # Apply attention threshold
        neighbor_attn_weights = torch.where(
            torch.abs(neighbor_attn_weights) >= self.attention_threshold,
            neighbor_attn_weights,
            torch.zeros_like(neighbor_attn_weights)
        )
        self_attn_weights = torch.where(
            torch.abs(self_attn_weights) >= self.attention_threshold,
            self_attn_weights,
            torch.zeros_like(self_attn_weights)
        )

        # Compute context vector
        context_vector = self.beta * (self_attn_weights * target_cell) + \
                         (1 - self.beta) * torch.sum(neighbor_attn_weights * neighbors, dim=1)

        # Compute reduced embedding
        reduced_embedding = F.leaky_relu(self.reduction_attn_layer(context_vector), negative_slope=0.01)

        return reduced_embedding, self_attn_weights, neighbor_attn_weights

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class GraphAttentionEncoder(nn.Module):
    def __init__(self, gene_dim, z_dim, radius, dropout_rate=0.3, beta_1=0.2, beta_2=0.8, alpha=2, attention_threshold=0.01):
        super(GraphAttentionEncoder, self).__init__()
        self.radius = radius
        self.gene_dim = gene_dim
        self.z_dim = z_dim
        self.dropout_rate = dropout_rate
        self.alpha = alpha
        self.attention_threshold = attention_threshold

        # Initialize attention layers
        self.attention1 = SelfNeighborAttention(gene_dim, gene_dim, radius, beta_1, alpha, attention_threshold)
        self.attention2 = SelfNeighborAttention(gene_dim, gene_dim, radius, beta_2, alpha, attention_threshold)

        # Final reduction layer
        self.reduction_matrix = nn.Linear(gene_dim, z_dim)

        self._initialize_weights()

        # For storing attention weights
        self.self_attention_weights1 = None
        self.self_attention_weights2 = None
        self.neighbor_attention_weights1 = None
        self.neighbor_attention_weights2 = None
        self.neighbor_indices = None

    def forward(self, x, edge_index, spatial_coords):
        x_agg, padded_neighbors, distances, neighbor_indices = self.aggregate_features(x, edge_index, spatial_coords)

        # Attention layer 1
        x_agg1, self.self_attention_weights1, self.neighbor_attention_weights1 = self.attention1(x_agg, padded_neighbors, distances)
        x_agg1 = F.leaky_relu(x_agg1, negative_slope=0.01)

        # Attention layer 2
        x_agg2, self.self_attention_weights2, self.neighbor_attention_weights2 = self.attention2(x_agg1, padded_neighbors, distances)
        x_agg2 = F.leaky_relu(x_agg2, negative_slope=0.01)

        # Final reduction
        x_reduced = F.leaky_relu(self.reduction_matrix(x_agg2), negative_slope=0.01)
        x_reduced = F.dropout(x_reduced, p=self.dropout_rate, training=self.training)

        self.neighbor_indices = neighbor_indices

        return x_reduced

    def aggregate_features(self, x, edge_index, spatial_coords):
        row, col = edge_index
        neighbors = x[col]

        aggregated_features = []
        neighbor_list = []
        distances_list = []
        neighbor_indices_list = []

        for node in range(x.size(0)):
            node_neighbors = neighbors[row == node]
            node_coords = spatial_coords[node]
            neighbor_coords = spatial_coords[col[row == node]]
            node_distances = torch.norm(node_coords - neighbor_coords, dim=1)

            if node_neighbors.size(0) > 0:
                aggregated = x[node]
                neighbor_indices = col[row == node].tolist()
            else:
                aggregated = x[node]
                neighbor_indices = []

            aggregated_features.append(aggregated)
            neighbor_list.append(node_neighbors)
            distances_list.append(node_distances)
            neighbor_indices_list.append(neighbor_indices)

        aggregated_features = torch.stack(aggregated_features)
        distances = torch.nn.utils.rnn.pad_sequence(distances_list, batch_first=True, padding_value=0.0)

        max_len = max(len(n) for n in neighbor_list)
        padded_neighbors = torch.zeros((x.size(0), max_len, x.size(1)), device=x.device)
        for i, n in enumerate(neighbor_list):
            if len(n) > 0:
                padded_neighbors[i, :len(n), :] = n

        return aggregated_features, padded_neighbors, distances, neighbor_indices_list

    def get_attention_info(self):
        return (
            self.self_attention_weights1,
            self.self_attention_weights2,
            self.neighbor_attention_weights1,
            self.neighbor_attention_weights2,
            self.neighbor_indices,
            self.reduction_matrix,
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)