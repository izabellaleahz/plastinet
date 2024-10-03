# plastinet/models/plastinet_model.py
from torch_geometric.data import DataLoader
from torch_geometric.nn import DeepGraphInfomax
import random
import numpy as np
import torch

from .attention import GraphAttentionEncoder
# from data.data_loader import create_data_objects

class PlastiNet:
    def __init__(self, adata, sample_key, radius, spatial_reg=0.2, z_dim=50, lr=0.005, beta=0.2, dropout=0.2, gamma=0.2, weight_decay=5e-4, epochs=30, random_seed=42):
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
        edge_subset_sz = 10000
        
        encoder = GraphAttentionEncoder(self.adata.shape[1], self.z_dim, self.radius, dropout_rate=self.dropout, beta=self.beta)

        self.model = DeepGraphInfomax(
            hidden_channels=self.z_dim,
            encoder=encoder,
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=lambda x, edge_index: (x[torch.randperm(x.size(0))], edge_index)
        ).to(self.device)

        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=self.gamma)
        best_params = self.model.state_dict()
        
        for epoch in range(self.epochs):
            train_loss = 0.0
            for batch in dataloader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                z, _, summary = self.model(batch.x, batch.edge_index)
                corrupted_z, corrupted_edge_index = self.model.corruption(z, batch.edge_index)
                loss = self.model.loss(z, corrupted_z, summary)
                coords = torch.tensor(batch.pos).float().to(self.device)
    
                # Calculate the distances
                cell_random_subset_1, cell_random_subset_2 = torch.randint(0, z.shape[0], (edge_subset_sz,)).to(
                    self.device), torch.randint(0, z.shape[0], (edge_subset_sz,)).to(self.device)
                z1, z2 = torch.index_select(z, 0, cell_random_subset_1), torch.index_select(z, 0, cell_random_subset_2)
                c1, c2 = torch.index_select(coords, 0, cell_random_subset_1), torch.index_select(coords, 0, cell_random_subset_2)  # Corrected subset selection
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

                loss = loss + self.spatial_reg * penalty_1 + 0.2 * l1_reg
                
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            self.epoch_losses.append(train_loss)
            scheduler.step()
    
            if epoch % 15 == 0:
                print(f"Epoch {epoch}/{self.epochs}, Loss: {train_loss}")
        
        self.model.load_state_dict(best_params)
        return


    def run_gat(self):
        data_list = create_data_objects(adata, self.sample_key)
        dataloader = DataLoader(data_list, batch_size=1, shuffle=True)

        self.train(dataloader)
        embeddings, cell_ids, self_attn_weights1, self_attn_weights2, attn_weights1, attn_weights2, neighbor_indices = self.generate_embeddings_with_ids(dataloader)

        embedding_adata = anndata.AnnData(embeddings)
        flattened_cell_ids = [item for sublist in cell_ids for item in sublist]
        embedding_adata.obs.index = flattened_cell_ids

        embedding_adata.obs = embedding_adata.obs.join(self.adata.obs)

        self.embedding_adata = embedding_adata 
        self.attn_weights1 = attn_weights1 
        self.attn_weights2 = attn_weights2 
        self.self_attn_weights1 = self_attn_weights1 
        self.self_attn_weights2 = self_attn_weights2 
        self.neighbor_indices = neighbor_indices
        
        return

    def get_embedding_adata(model, dataloader, device):
        model.eval()
        embeddings = []
        cell_ids = []
        for batch in dataloader:
            batch = batch.to(device)
            with torch.no_grad():
                z = model.encoder(batch.x, batch.edge_index)
                embeddings.append(z.cpu().numpy())
                cell_ids.extend(batch.cell_ids)
        embeddings = np.concatenate(embeddings, axis=0)
        
        embedding_adata = anndata.AnnData(embeddings)
        flattened_cell_ids = [item for sublist in cell_ids for item in sublist]
        embedding_adata.obs.index = flattened_cell_ids
        
        embedding_adata.obs = embedding_adata.obs.join(adata.obs)
        return embedding_adata 
