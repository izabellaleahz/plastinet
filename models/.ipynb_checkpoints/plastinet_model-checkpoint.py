from torch_geometric.data import DataLoader
from torch_geometric.nn import DeepGraphInfomax
import random
import numpy as np
import torch
import anndata

from .attention import GraphAttentionEncoder
from ..data.data_loader import create_data_objects
from ..visualization.plots import plot_graph
from ..analysis.attention_analysis import get_gatt, prep_for_gatt

class PlastiNet:
    def __init__(self, adata, sample_key, radius, spatial_reg=0.2, z_dim=50, lr=0.005, beta=0.2, dropout=0, gamma=0.5, weight_decay=0, epochs=30, random_seed=42):
        self.adata = adata
        self.sample_key = sample_key
        self.radius = radius
        self.spatial_reg = spatial_reg
        self.z_dim = z_dim
        self.lr = lr
        self.beta = beta
        self.dropout = dropout
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.random_seed = random_seed
        self.device = 'cpu'
        self.model = None
        self.epoch_losses = []

    def train(self, dataloader):
        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        # TODO: Make this a percentage that is passed in so for ex 0.2 is 20% of all the input cells 
        edge_subset_sz = 10000

        encoder = GraphAttentionEncoder(self.adata.shape[1], self.z_dim, self.radius, dropout_rate=self.dropout, beta=self.beta)

        self.model = DeepGraphInfomax(
            hidden_channels=self.z_dim,
            encoder=encoder,
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=lambda x, edge_index, pos: (x[torch.randperm(x.size(0))], edge_index, pos)
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=self.gamma)
        best_params = self.model.state_dict()
        
        for epoch in range(self.epochs):
            train_loss = 0.0
            for batch in dataloader:
                batch = batch.to(self.device)

                assert batch.pos is not None, "batch.pos is missing!"
                assert batch.pos.size(0) == batch.x.size(0), f"Shape mismatch between batch.pos ({batch.pos.size()}) and batch.x ({batch.x.size()})"

                optimizer.zero_grad()
                z, _, summary = self.model(batch.x, batch.edge_index, batch.pos)
                corrupted_z, corrupted_edge_index, _ = self.model.corruption(z, batch.edge_index, batch.pos)
                loss = self.model.loss(z, corrupted_z, summary)
                coords = torch.tensor(batch.pos).float().to(self.device)
    
                # Calculate the distances
                cell_random_subset_1, cell_random_subset_2 = torch.randint(0, z.shape[0], (edge_subset_sz,)).to(self.device), torch.randint(0, z.shape[0], (edge_subset_sz,)).to(self.device)
                z1, z2 = torch.index_select(z, 0, cell_random_subset_1), torch.index_select(z, 0, cell_random_subset_2)
                c1, c2 = torch.index_select(coords, 0, cell_random_subset_1), torch.index_select(coords, 0, cell_random_subset_2)
                pdist = torch.nn.PairwiseDistance(p=2)
                
                z_dists = pdist(z1, z2)
                z_dists = z_dists / torch.max(z_dists)
                
                sp_dists = pdist(c1, c2)
                sp_dists = sp_dists / torch.max(sp_dists)
                n_items = z_dists.size(dim=0)
                
                penalty_1 = torch.div(torch.sum(torch.mul(1.0 - z_dists, sp_dists)), n_items).to(self.device)
                
                l1_reg = 0
                for name, param in self.model.encoder.named_parameters():
                    if 'self_attn_layer' in name or 'neighbor_attn_layer' in name:
                        l1_reg += torch.norm(param, p=1)
                print("DGI: ", 0.5*loss)
                print("spatial: ", penalty_1)
                print("L1: ", 0.00001 * l1_reg)
                loss = 0.5 * loss + penalty_1 + 0.00001 * l1_reg
                
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            self.epoch_losses.append(train_loss)
            scheduler.step()

            if epoch % 1 == 0:
                print(f"Epoch {epoch}/{self.epochs}, Loss: {train_loss}")
        
        self.model.load_state_dict(best_params)
        
        return

    def run_gat(self):
        data_list = create_data_objects(self.adata, self.sample_key, self.radius)
        for graph in data_list:
            plot_graph(graph)
        dataloader = DataLoader(data_list, batch_size=1, shuffle=True)
    
        self.train(dataloader)
    
        #generate an embedding_adata
        self.generate_embedding_adata(dataloader)

        return

    def generate_embedding_adata(self, dataloader):
        '''Creates and saves the embedding adata object and attention weights.'''
        self.model.eval()  
        
        embeddings, cell_ids, self_attn_weights, neighbor_attn_weights, reduction_attn_weights, neighbor_indices = [], [], [], [], [], []
        max_neighbors = 0
        for batch in dataloader:
            batch = batch.to(self.device)
            with torch.no_grad():
                z = self.model.encoder(batch.x, batch.edge_index, batch.pos)
                embeddings.append(z.cpu().numpy())
                cell_ids.extend(batch.cell_ids)

                # Retrieve and detach attention-related outputs
                self_attn, neighbor_attn, reduction_attn, neighbors = self.model.encoder.get_attention_info()

                self_attn_weights.append(self_attn.detach().cpu().numpy())
                neighbor_attn_weights.append(neighbor_attn.detach().cpu().numpy())
                reduction_attn_weights.append(reduction_attn.detach().cpu().numpy())
                neighbors_tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor(n, dtype=torch.long, device=self.device) for n in neighbors], batch_first=True, padding_value=-1)
                neighbor_indices.append(neighbors_tensor.cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        self_attn_weights = np.concatenate(self_attn_weights, axis=0)
        neighbor_attn_weights = np.concatenate(neighbor_attn_weights, axis=0)
        reduction_attn_weights = np.concatenate(reduction_attn_weights, axis=0)
        neighbor_indices = np.concatenate(neighbor_indices, axis=0)
            # Create an AnnData object for embeddings
        embedding_adata = anndata.AnnData(embeddings)
        embedding_adata.obs.index = cell_ids
        embedding_adata.obs = self.adata.obs
       
        self.embedding_adata = embedding_adata
        self.self_attn_weights = self_attn_weights
        self.neighbor_attn_weights = neighbor_attn_weights
        self.reduction_attn_weights = reduction_attn_weights
        self.neighbor_indices = neighbor_indices

        return embedding_adata