import torch.nn as nn
import torch.nn.functional as F
import torch 

class SelfNeighborAttention(nn.Module):
    def __init__(self, gene_dim, radius, beta, alpha, attention_threshold=0.01):
        super(SelfNeighborAttention, self).__init__()
        self.gene_dim = gene_dim
        self.beta = beta
        self.alpha = alpha
        self.radius = radius 
        self.attention_threshold = attention_threshold
        
        self.layer_norm = nn.LayerNorm(gene_dim)
        self.self_attn_layer = nn.Linear(gene_dim, gene_dim)
        self.neighbor_attn_layer = nn.Linear(gene_dim, gene_dim)
        
        self._initialize_weights()

    def forward(self, target_cell, neighbors, distances, neighbor_indices):
        # Normalize
        target_cell = self.layer_norm(target_cell)
        neighbors = self.layer_norm(neighbors)

        # Compute attention scores
        self_attn_scores = self.self_attn_layer(target_cell)  # [B, gene_dim]
        neighbor_attn_scores = self.neighbor_attn_layer(neighbors)  # [B, N, gene_dim]

        # Compute distance weights
        normalized_distances = distances / (self.radius + 1e-8)
        distance_weights = torch.exp(-self.alpha * normalized_distances)  # [B, N]
        weighted_neighbor_scores = neighbor_attn_scores * distance_weights.unsqueeze(-1)  # [B, N, gene_dim]

        # Combine self and neighbor scores
        combined_scores = torch.cat([self_attn_scores.unsqueeze(1), weighted_neighbor_scores], dim=1)  # [B, 1+N, gene_dim]

        # Create a mask for valid neighbors: True = valid, False = padded
        valid_mask = (neighbor_indices != -1)  # [B, N]
        # We need to account for self position as well. Self is always valid
        # So we will create a mask that includes self at the front:
        # Shape: [B, 1+N], first column is True for self, then valid_mask for neighbors
        full_mask = torch.cat([torch.ones((neighbors.size(0), 1), dtype=torch.bool, device=neighbors.device), valid_mask], dim=1)

        # Mask invalid entries before softmax
        # Set masked positions to a large negative value so softmax ~0
        combined_scores = combined_scores.masked_fill(~full_mask.unsqueeze(-1), float('-inf'))

        # Apply softmax over neighbors dimension
        combined_weights = F.softmax(combined_scores, dim=1)

        # Extract self and neighbor attention weights
        self_attn_weights = combined_weights[:, 0, :]  # [B, gene_dim]
        neighbor_attn_weights = combined_weights[:, 1:, :]  # [B, N, gene_dim]

        # Compute context vector
        # Weighted combination of self and neighbor features
        # Only valid neighbors contribute due to the masking in weights
        context_vector = self.beta * (self_attn_weights * target_cell) \
                         + (1 - self.beta) * torch.sum(neighbor_attn_weights * neighbors, dim=1)

        return context_vector, self_attn_weights, neighbor_attn_weights

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class GraphAttentionEncoder(nn.Module):
    def __init__(self, gene_dim, z_dim, radius, dropout_rate, beta_1, beta_2, alpha, attention_threshold):
        super(GraphAttentionEncoder, self).__init__()
        self.radius = radius
        self.gene_dim = gene_dim
        self.z_dim = z_dim
        self.dropout_rate = dropout_rate
        self.attention_threshold = attention_threshold 

        self.attention1 = SelfNeighborAttention(gene_dim, radius, beta_1, alpha, self.attention_threshold)
        self.attention2 = SelfNeighborAttention(gene_dim, radius, beta_2, alpha, self.attention_threshold)
        
        self.reduction_matrix = nn.Linear(gene_dim, z_dim)

        self.ffn = nn.Sequential(
            nn.Linear(gene_dim, 2 * gene_dim), 
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Linear(2 * gene_dim, z_dim),    
            nn.LeakyReLU(negative_slope=0.01) 
        )
            
        self._initialize_weights()
        
        self.self_attention_weights1 = None
        self.self_attention_weights2 = None
        self.neighbor_attention_weights1 = None
        self.neighbor_attention_weights2 = None
        self.neighbor_indices = None

    def forward(self, x, edge_index, spatial_coords):
        x_agg, padded_neighbors, distances, padded_indices = self.aggregate_features(x, edge_index, spatial_coords)
    
        x_agg1, self.self_attention_weights1, self.neighbor_attention_weights1 = self.attention1(x_agg, padded_neighbors, distances, padded_indices)
        x_agg1 = F.leaky_relu(x_agg1, negative_slope=0.01)
        
        x_agg2, self.self_attention_weights2, self.neighbor_attention_weights2 = self.attention2(x_agg1, padded_neighbors, distances, padded_indices)
        x_agg2 = F.leaky_relu(x_agg2, negative_slope=0.01)

        x_reduced = self.reduction_matrix(x_agg2)

        self.neighbor_indices = padded_indices  

        return x_reduced  
        
    def aggregate_features(self, x, edge_index, spatial_coords):
        row, col = edge_index
        neighbors = x[col].view(-1, x.size(1))

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
                aggregated = torch.sum(node_neighbors, dim=0)
                n_indices = col[row == node].tolist()  
            else:
                aggregated = torch.zeros(x.size(1), device=x.device)
                n_indices = []

            aggregated_features.append(aggregated)
            neighbor_list.append(node_neighbors)
            distances_list.append(node_distances)
            neighbor_indices_list.append(n_indices)  

        aggregated_features = torch.stack(aggregated_features)
        distances = torch.nn.utils.rnn.pad_sequence(distances_list, batch_first=True, padding_value=0.0)

        max_len = max(len(n) for n in neighbor_list)
        padded_neighbors = torch.zeros((x.size(0), max_len, x.size(1)), device=x.device)
        padded_indices = torch.full((x.size(0), max_len), -1, dtype=torch.long, device=x.device)

        for i, n in enumerate(neighbor_list):
            padded_neighbors[i, :len(n), :] = n
            if len(neighbor_indices_list[i]) > 0:
                padded_indices[i, :len(neighbor_indices_list[i])] = torch.tensor(neighbor_indices_list[i], dtype=torch.long, device=x.device)

        return aggregated_features, padded_neighbors, distances, padded_indices

    def get_attention_info(self):
        return (self.self_attention_weights1,
                self.self_attention_weights2,
                self.neighbor_attention_weights1,
                self.neighbor_attention_weights2,
                self.neighbor_indices,
                self.reduction_matrix)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
