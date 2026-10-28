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

    def forward(self, target_cell, neighbors, distances, neighbor_mask=None):
        target_cell = self.layer_norm(target_cell)
        neighbors = self.layer_norm(neighbors)

        self_attn_scores = self.self_attn_layer(target_cell)
        neighbor_attn_scores = self.neighbor_attn_layer(neighbors)
    
        # Additive log-space penalty: equivalent to scaling each neighbor's unnormalized weight by exp(-alpha * d),
        # so nearer neighbors are always favored regardless of the sign of the score
        normalized_distances = distances / (self.radius + 1e-8)
        weighted_neighbor_scores = neighbor_attn_scores - self.alpha * normalized_distances.unsqueeze(-1)

        # Padded neighbor slots must receive zero attention
        if neighbor_mask is not None:
            weighted_neighbor_scores = weighted_neighbor_scores.masked_fill(~neighbor_mask.unsqueeze(-1), float('-inf'))
    
        combined_scores = torch.cat([self_attn_scores.unsqueeze(1), weighted_neighbor_scores], dim=1)
        combined_weights = F.softmax(combined_scores, dim=1)

        self_attn_weights = combined_weights[:, 0, :]
        neighbor_attn_weights = combined_weights[:, 1:, :]

        context_vector = self.beta * (self_attn_weights * target_cell) + (1 - self.beta) * torch.sum(neighbor_attn_weights * neighbors, dim=1)
    
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
        padded_neighbors, distances, neighbor_mask, neighbor_indices = self.aggregate_features(x, edge_index, spatial_coords)
    
        # The cell's own expression is the self term; neighbors enter through the neighbor attention
        x_agg1, self.self_attention_weights1, self.neighbor_attention_weights1 = self.attention1(x, padded_neighbors, distances, neighbor_mask)
        x_agg1 = F.leaky_relu(x_agg1, negative_slope=0.01)
        
        x_agg2, self.self_attention_weights2, self.neighbor_attention_weights2 = self.attention2(x_agg1, padded_neighbors, distances, neighbor_mask)
        x_agg2 = F.leaky_relu(x_agg2, negative_slope=0.01)

        x_reduced = self.reduction_matrix(x_agg2)

        # x_reduced = self.ffn(x_agg2)
        
        # x_reduced = F.leaky_relu(x_reduced, negative_slope=0.01) 
        # x_reduced = F.dropout(x_reduced, p=self.dropout_rate, training=self.training)

        self.neighbor_indices = neighbor_indices  

        return x_reduced  
        
    def aggregate_features(self, x, edge_index, spatial_coords):
        # Pack each node's neighbors into a padded (n_nodes, max_degree, n_genes) tensor, keeping edge order within a node
        n_nodes = x.size(0)
        row, col = edge_index
        order = torch.argsort(row, stable=True)
        row, col = row[order], col[order]

        degree = torch.bincount(row, minlength=n_nodes)
        max_len = int(degree.max()) if row.numel() > 0 else 0
        slot = torch.arange(row.numel(), device=x.device) - (torch.cumsum(degree, 0) - degree)[row]

        padded_neighbors = x.new_zeros((n_nodes, max_len, x.size(1)))
        padded_neighbors[row, slot] = x[col]

        distances = x.new_zeros((n_nodes, max_len))
        distances[row, slot] = torch.norm(spatial_coords[row] - spatial_coords[col], dim=1)

        neighbor_mask = torch.zeros((n_nodes, max_len), dtype=torch.bool, device=x.device)
        neighbor_mask[row, slot] = True

        neighbor_indices_list = [c.tolist() for c in torch.split(col, degree.tolist())]

        return padded_neighbors, distances, neighbor_mask, neighbor_indices_list

    def get_attention_info(self):
        return self.self_attention_weights1, self.self_attention_weights2, self.neighbor_attention_weights1, self.neighbor_attention_weights2, self.neighbor_indices, self.reduction_matrix

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)