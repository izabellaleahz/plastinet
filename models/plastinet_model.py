# plastinet/models/plastinet_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import anndata
import scanpy as sc
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DeepGraphInfomax
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from .attention import GraphAttentionEncoder
from ..data.data_loader import create_data_objects



class PlastiNet:
    def __init__(
        self,
        adata,
        sample_key,
        radius,
        spatial_reg=0.2,
        l1_reg=1e-5,
        z_dim=50,
        lr=0.001,
        beta_1=0.2,
        beta_2=0.8,
        alpha=3,
        attention_threshold=0.01,
        dropout=0.2,
        gamma=0.8,
        weight_decay=0.005,
        epochs=80,
        random_seed=42,
        patience=10,
        mask_n=0.7,
        spatial_percent=0.2,
        step_size=5,
    ):
        self.adata = adata
        self.sample_key = sample_key
        self.radius = radius
        self.spatial_reg = spatial_reg
        self.z_dim = z_dim
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.alpha = alpha
        self.attention_threshold = attention_threshold
        self.dropout = dropout
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.random_seed = random_seed
        self.patience = patience
        self.mask_n = mask_n
        self.spatial_percent = spatial_percent
        self.step_size = step_size
        self.l1_reg = l1_reg

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        import random

        random.seed(self.random_seed)

    def train(self, dataloader):
        # Initialize the encoder
        encoder = GraphAttentionEncoder(
            self.adata.shape[1],
            self.z_dim,
            self.radius,
            dropout_rate=self.dropout,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            alpha=self.alpha,
            attention_threshold=self.attention_threshold,
        ).to(self.device)

        self.model = DeepGraphInfomax(
            hidden_channels=self.z_dim,
            encoder=encoder,
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=lambda x, edge_index, pos: (
                x * torch.bernoulli(torch.ones_like(x) * self.mask_n),
                edge_index,
                pos,
            ),
        ).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

        best_metric = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            for batch in dataloader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                pos_z, _, summary = self.model(batch.x, batch.edge_index, batch.pos)
                neg_z, _, _ = self.model.corruption(batch.x, batch.edge_index, batch.pos)

                # DGI Loss
                dgi_loss = self.model.loss(pos_z, neg_z, summary)

                # Spatial Loss
                spatial_loss = self.compute_spatial_loss(pos_z, batch.pos, self.spatial_percent)

                # L1 Regularization Loss
                l1_loss = sum(
                    torch.norm(param, p=1)
                    for name, param in self.model.encoder.named_parameters()
                    if "attn" in name and param.requires_grad
                )
                l1_loss *= self.l1_reg

                # Total Loss
                total_loss = dgi_loss + spatial_loss * self.spatial_reg + l1_loss
                total_loss.backward()
                optimizer.step()
                train_loss += total_loss.item()

            scheduler.step()

            if train_loss < best_metric:
                best_metric = train_loss
                patience_counter = 0
                best_params = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        self.model.load_state_dict(best_params)

    def compute_spatial_loss(self, z, coords, subset_percent=0.2):
        edge_subset_sz = int(subset_percent * z.shape[0])

        cell_random_subset_1 = torch.randint(0, z.size(0), (edge_subset_sz,)).to(self.device)
        cell_random_subset_2 = torch.randint(0, z.size(0), (edge_subset_sz,)).to(self.device)

        z1, z2 = z[cell_random_subset_1], z[cell_random_subset_2]
        c1, c2 = coords[cell_random_subset_1], coords[cell_random_subset_2]

        z_dists = torch.norm(z1 - z2, dim=1) / (torch.max(torch.norm(z1 - z2, dim=1)) + 1e-8)
        sp_dists = torch.norm(c1 - c2, dim=1) / (torch.max(torch.norm(c1 - c2, dim=1)) + 1e-8)

        spatial_loss = torch.sum((1.0 - z_dists) * sp_dists) / len(z_dists)
        return spatial_loss

    def run_gat(self):
        print("Starting GAT run...")
        data_list = create_data_objects(self.adata, self.sample_key, self.radius)
        dataloader = DataLoader(data_list, batch_size=1, shuffle=True)

        self.train(dataloader)

        embedding_adata = self.generate_embedding_adata(dataloader)
        print("GAT run completed.")
        return embedding_adata
        
    def generate_embedding_adata(self, dataloader):
        """
        Generate an embedding AnnData object with attention information and reduction matrix.
        """
        self.model.eval()
        embeddings = []
        cell_ids = []
        self_attn_weights1 = []
        self_attn_weights2 = []
        neighbor_attn_weights1 = []
        neighbor_attn_weights2 = []
        neighbor_indices = []
        reduction_matrix = None
    
        for batch in dataloader:
            batch = batch.to(self.device)
            with torch.no_grad():
                # Use the encoder to generate embeddings
                z = self.model.encoder(batch.x, batch.edge_index, batch.pos)
                embeddings.append(z.cpu().numpy())
                cell_ids.extend(batch.cell_ids)  # Assuming `cell_ids` exist in batch
    
                # Extract attention weights, neighbor indices, and reduction matrix
                (
                    self_attn1,
                    self_attn2,
                    neighbor_attn1,
                    neighbor_attn2,
                    neighbors,
                    reduction_layer,
                ) = self.model.encoder.get_attention_info()
    
                # Save attention weights and neighbor indices
                self_attn_weights1.append(self_attn1.cpu().numpy())
                self_attn_weights2.append(self_attn2.cpu().numpy())
                neighbor_attn_weights1.append(
                    [attn.cpu().numpy() for attn in neighbor_attn1]
                )
                neighbor_attn_weights2.append(
                    [attn.cpu().numpy() for attn in neighbor_attn2]
                )
                neighbor_indices.append(
                    [torch.tensor(neigh, dtype=torch.long).cpu().numpy() for neigh in neighbors]
                )
    
                # Save the reduction matrix
                if reduction_matrix is None:
                    reduction_matrix = reduction_layer.weight.cpu().detach().numpy()
    
        # Concatenate embeddings
        embeddings = np.concatenate(embeddings, axis=0)
    
        # Create AnnData object
        embedding_adata = anndata.AnnData(embeddings, obs=self.adata.obs.copy())
        embedding_adata.obs.index = cell_ids  # Align with cell IDs
    
        # Save attention weights into `obsm`
        embedding_adata.obsm["self_attention_weights_layer1"] = np.concatenate(
            self_attn_weights1, axis=0
        )
        embedding_adata.obsm["self_attention_weights_layer2"] = np.concatenate(
            self_attn_weights2, axis=0
        )
    
        # Save neighbor attention weights and indices into `uns`
        embedding_adata.uns["neighbor_attention"] = {
            "layer1": neighbor_attn_weights1,
            "layer2": neighbor_attn_weights2,
            "indices": neighbor_indices,
        }
    
        # Save reduction matrix into `uns`
        embedding_adata.uns["reduction_matrix"] = reduction_matrix
    
        # Save embedding_adata to the class
        self.embedding_adata = embedding_adata
        return embedding_adata
