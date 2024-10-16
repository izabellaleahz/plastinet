import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 

class SelfNeighborAttention(nn.Module):
    def __init__(self, gene_dim, reduced_dim, radius, beta=0.5, attention_threshold=0.01):
        super(SelfNeighborAttention, self).__init__()
        self.gene_dim = gene_dim
        self.reduced_dim = reduced_dim
        self.beta = nn.Parameter(torch.tensor(beta)) 
        self.radius = radius 
        self.attention_threshold = attention_threshold
        
        self.layer_norm = nn.LayerNorm(gene_dim)
        self.self_attn_layer = nn.Linear(gene_dim, gene_dim)
        self.neighbor_attn_layer = nn.Linear(gene_dim, gene_dim)
        
        self.reduction_attn_layer = nn.Linear(gene_dim, reduced_dim)
        self._initialize_weights()

    def forward(self, target_cell, neighbors, distances):
        
        target_cell = self.layer_norm(target_cell)
        neighbors = self.layer_norm(neighbors)
        
        self_attn_scores = self.self_attn_layer(target_cell)
        neighbor_attn_scores = self.neighbor_attn_layer(neighbors)
        

        normalized_distances = distances / (self.radius + 1e-8)
        alpha = 2
        distance_weights = torch.exp(-alpha * normalized_distances)
        weighted_neighbor_scores = neighbor_attn_scores * distance_weights.unsqueeze(-1)

        combined_scores = torch.cat([self_attn_scores.unsqueeze(1), weighted_neighbor_scores], dim=1)

        #yes do need this  
        combined_weights = F.softmax(combined_scores, dim=1)
        
        self_attn_weights = combined_weights[:, 0, :]
        
        neighbor_attn_weights = combined_weights[:, 1:, :]

        # plt.hist(self_attn_weights.detach().numpy())
        # plt.show()
        # print("neighbor: ", neighbor_attn_weights.detach().numpy())

        neighbor_attn_weights = torch.where(torch.abs(neighbor_attn_weights) >= self.attention_threshold, neighbor_attn_weights, torch.zeros_like(neighbor_attn_weights))

        self_attn_weights = torch.where(torch.abs(self_attn_weights) >= self.attention_threshold, self_attn_weights, torch.zeros_like(self_attn_weights))
        
        context_vector = self.beta * (self_attn_weights * target_cell) + (1 - self.beta) * torch.sum(neighbor_attn_weights * neighbors, dim=1)

        reduced_embedding = F.leaky_relu(self.reduction_attn_layer(context_vector), negative_slope=0.01)

        return reduced_embedding, self_attn_weights, neighbor_attn_weights

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class GraphAttentionEncoder(nn.Module):
    def __init__(self, gene_dim, reduced_dim, radius, dropout_rate=0.3, beta=0.5):
        super(GraphAttentionEncoder, self).__init__()
        self.radius = radius
        self.gene_dim = gene_dim
        self.reduced_dim = reduced_dim
       
        self.attention = SelfNeighborAttention(gene_dim, reduced_dim, radius, beta)
        self._initialize_weights()

        self.self_attention_weights = None
        self.neighbor_attention_weights = None
        self.reduction_attention_weights = None
        self.neighbor_indices = None

    def forward(self, x, edge_index, spatial_coords):
        x_agg, padded_neighbors, distances, neighbor_indices = self.aggregate_features(x, edge_index, spatial_coords)
        x_agg, self.self_attention_weights, self.neighbor_attention_weights = self.attention(x, padded_neighbors, distances)

        self.reduction_attention_weights = self.self_attention_weights
        self.neighbor_indices = neighbor_indices  
        x_agg = F.leaky_relu(x_agg, negative_slope=0.01)
        x_agg = F.dropout(x_agg, training=self.training)

        return x_agg
        
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
                neighbor_indices = col[row == node].tolist()  
            else:
                aggregated = torch.zeros(x.size(1), device=x.device)
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
            padded_neighbors[i, :len(n), :] = n

        return aggregated_features, padded_neighbors, distances, neighbor_indices_list

    def get_attention_info(self):
        return self.self_attention_weights, self.neighbor_attention_weights, self.reduction_attention_weights, self.neighbor_indices

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
