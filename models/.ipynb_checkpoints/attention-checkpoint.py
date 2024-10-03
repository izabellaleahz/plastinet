# plastinet/models/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfNeighborAttention(nn.Module):
    def __init__(self, gene_dim, reduced_dim, beta=0.2):
        super(SelfNeighborAttention, self).__init__()
        self.gene_dim = gene_dim
        self.reduced_dim = reduced_dim
        self.beta = beta
        
        self.self_attn_layer = nn.Linear(gene_dim, gene_dim)
        self.neighbor_attn_layer = nn.Linear(gene_dim, gene_dim)
        self.ffn = nn.Sequential(
            nn.Linear(gene_dim, gene_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(gene_dim, reduced_dim)
        )
        self._initialize_weights()
        
    def forward(self, target_cell, neighbors, distances):
        min_dist = 0
        max_dist = 250
        normalized_distances = (distances - min_dist) / (max_dist - min_dist + 1e-8)  # Adding epsilon to avoid division by zero
        # Self-attention scores
        self_attn_scores = self.self_attn_layer(target_cell)
        self_attn_weights = self_attn_scores
    
        # Neighbor attention scores
        neighbor_attn_scores = self.neighbor_attn_layer(neighbors)
        
        alpha=0.3
        distance_weights = torch.exp(-alpha * normalized_distances)
        weighted_neighbor_scores = neighbor_attn_scores * distance_weights.unsqueeze(-1)

        # linear decay
        # distance_weights = 1 - normalized_distances
        # distance_weights = torch.clamp(distance_weights, min=0)  # Ensure non-negative weights
        # weighted_neighbor_scores = neighbor_attn_scores * distance_weights.unsqueeze(-1)

    
        # Normalize the weighted neighbor scores using softmax to ensure smooth transition
        neighbor_attn_weights = F.softmax(weighted_neighbor_scores, dim=1)
        # neighbor_attn_output = torch.sum(neighbor_attn_weights * neighbors, dim=1)
        neighbor_attn_output = torch.sum(neighbor_attn_weights * neighbors, dim=1)
        
        context_vector = self.beta * (self_attn_weights * target_cell) + (1 - self.beta) * neighbor_attn_output
        
        reduced_embedding = self.ffn(context_vector)  
        
        return reduced_embedding, self_attn_weights, neighbor_attn_weights


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class GraphAttentionEncoder(nn.Module):
    def __init__(self, gene_dim, reduced_dim, radius, dropout_rate=0.3, beta=0.2):
        super(GraphAttentionEncoder, self).__init__()
        self.radius = radius
        self.gene_dim = gene_dim
        self.reduced_dim = reduced_dim
        self.attention1 = SelfNeighborAttention(gene_dim, gene_dim, beta)
        self.attention2 = SelfNeighborAttention(gene_dim, reduced_dim, beta)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention_weights1 = None
        self.attention_weights2 = None
        self.self_attention_weights1 = None
        self.self_attention_weights2 = None
        self.neighbor_indices = None
        self._initialize_weights()

    def forward(self, x, edge_index):
        x_agg, padded_neighbors, neighbor_counts, neighbor_indices, distances = self.aggregate_features(x, edge_index)
        self.neighbor_indices = neighbor_indices
        
        x_agg, self.self_attention_weights1, self.attention_weights1 = self.attention1(x, padded_neighbors, distances)
        x_agg = F.relu(x_agg)
        x_agg = self.dropout(x_agg)
        
        x_agg, self.self_attention_weights2, self.attention_weights2 = self.attention2(x_agg, padded_neighbors, distances)
        x_agg = F.relu(x_agg)
        
        return x_agg

    def aggregate_features(self, x, edge_index):
        row, col = edge_index
        neighbors = x[col].view(-1, x.size(1))
        
        aggregated_features = []
        neighbor_list = []
        neighbor_counts = []
        neighbor_indices_list = []
        distances_list = []

        for node in range(x.size(0)):
            node_neighbors = neighbors[row == node]
            node_distances = torch.norm(x[node] - node_neighbors, dim=1)
            
            if node_neighbors.size(0) > 0:
                aggregated = torch.sum(node_neighbors, dim=0)
                neighbor_count = node_neighbors.size(0)
            else:
                aggregated = torch.zeros(x.size(1), device=x.device)
                neighbor_count = 0
            aggregated_features.append(aggregated)
            neighbor_list.append(node_neighbors)
            neighbor_counts.append(neighbor_count)
            neighbor_indices_list.append(col[row == node].tolist())
            distances_list.append(node_distances)
        
        aggregated_features = torch.stack(aggregated_features)
        neighbor_counts = torch.tensor(neighbor_counts, device=x.device)
        distances = torch.nn.utils.rnn.pad_sequence(distances_list, batch_first=True, padding_value=0.0)
        
        max_len = max(len(n) for n in neighbor_list)
        padded_neighbors = torch.zeros((x.size(0), max_len, x.size(1)), device=x.device)
        for i, n in enumerate(neighbor_list):
            padded_neighbors[i, :len(n), :] = n
        
        return aggregated_features, padded_neighbors, neighbor_counts, neighbor_indices_list, distances

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def get_attention_info(self):
        return self.self_attention_weights1, self.self_attention_weights2, self.attention_weights1, self.attention_weights2, self.neighbor_indices
