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
        l1_reg = 1e-5, 
        z_dim=50,
        lr=0.001,
        beta_1= 0.2,
        beta_2 = 0.8,
        alpha=3,
        attention_threshold = 0.01,
        dropout=0.2,
        gamma=0.8,
        weight_decay=0.005,
        epochs=80,
        random_seed=42,
        patience=10,
        mask_n=0.7,
        spatial_percent=0.2,
        step_size=5
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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.cluster_centers = None

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        import random
        random.seed(self.random_seed)

    def train(self, dataloader):
        # Initialize the encoder with float alpha and beta
        encoder = GraphAttentionEncoder(
            self.adata.shape[1],
            self.z_dim,
            self.radius,
            dropout_rate=self.dropout,
            beta_1 = self.beta_1,
            beta_2 = self.beta_2,
            alpha=self.alpha,
            attention_threshold = self.attention_threshold
        ).to(self.device)

        self.model = DeepGraphInfomax(
            hidden_channels=self.z_dim,
            encoder=encoder,
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=lambda x, edge_index, pos: (
                x * torch.bernoulli(torch.ones_like(x) * self.mask_n),
                edge_index,
                pos
            )
        ).to(self.device)

        

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # Use StepLR scheduler
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.step_size,
            gamma=self.gamma
        )

        best_metric = float('inf')
        patience_counter = 0

        dgi_loss_list = []
        l1_loss_list = []
        spatial_loss_list = []
        total_loss_list = []

        for epoch in range(self.epochs):
            self.model.train()  # Set model to training mode
            train_loss = 0.0
            epoch_dgi_loss = 0.0
            epoch_l1_loss = 0.0
            epoch_spatial_loss = 0.0

            for batch in dataloader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                # Forward pass with internal corruption handling
                pos_z, neg_z, summary= self.model(batch.x, batch.edge_index, batch.pos)
                
                # neg_z, _, _ = self.model.corruption(pos_z, batch.edge_index, batch.pos)

                
                # DGI Loss
                dgi_loss = self.model.loss(pos_z, neg_z, summary)

                # Spatial Loss
                spatial_loss = self.compute_spatial_loss(pos_z, batch.pos, self.spatial_percent)

                # L1 Regularization Loss
                l1_loss = 0.0
                for name, param in self.model.encoder.named_parameters():
                    if 'attn' in name and param.requires_grad:
                        l1_loss += torch.norm(param, p=1)
                l1_loss = self.l1_reg * l1_loss
                spatial_loss = self.spatial_reg * spatial_loss
                # Total Loss
                total_loss = dgi_loss + spatial_loss + l1_loss

                # Backward pass and optimization
                total_loss.backward()
                optimizer.step()

                # Accumulate losses
                train_loss += total_loss.item()
                epoch_dgi_loss += dgi_loss.item()
                epoch_l1_loss += l1_loss.item()
                epoch_spatial_loss += spatial_loss.item()

            # Step the scheduler
            scheduler.step()

            # Early stopping logic
            if train_loss < best_metric:
                best_metric = train_loss
                patience_counter = 0
                best_params = self.model.state_dict()
                print("Model improved and parameters saved.")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

            # Append epoch losses for plotting
            dgi_loss_list.append(epoch_dgi_loss)
            l1_loss_list.append(epoch_l1_loss)
            spatial_loss_list.append(epoch_spatial_loss)
            total_loss_list.append(train_loss)

            # Log the losses at the end of each epoch
            print(f"Epoch [{epoch+1}/{self.epochs}] Completed. "
                  f"Epoch Losses: DGI Loss = {epoch_dgi_loss:.4f}, "
                  f"Spatial Loss = {epoch_spatial_loss:.4f}, "
                  f"L1 Loss = {epoch_l1_loss:.6f}, "
                  f"Total Loss = {train_loss:.4f}")

        # Load the best model parameters
        self.model.load_state_dict(best_params)

        # Plot the loss trends over epochs
        epochs_range = range(1, len(dgi_loss_list) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, dgi_loss_list, label='DGI Loss', marker='o')
        plt.plot(epochs_range, spatial_loss_list, label='Spatial Loss', marker='o')
        plt.plot(epochs_range, l1_loss_list, label='L1 Loss', marker='o')
        plt.plot(epochs_range, total_loss_list, label='Total Loss', marker='o')

        plt.title('Loss Trends Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.show()

      
    def compute_spatial_loss(self, z, coords, subset_percent=0.2):
        # Compute spatial loss to enforce spatial awareness
        edge_subset_sz = int(subset_percent * z.shape[0])

        cell_random_subset_1 = torch.randint(0, z.size(0), (edge_subset_sz,)).to(self.device)
        cell_random_subset_2 = torch.randint(0, z.size(0), (edge_subset_sz,)).to(self.device)

        z1 = z[cell_random_subset_1]
        z2 = z[cell_random_subset_2]
        c1 = coords[cell_random_subset_1]
        c2 = coords[cell_random_subset_2]

        z_dists = torch.norm(z1 - z2, dim=1)
        z_dists = z_dists / (torch.max(z_dists) + 1e-8)

        sp_dists = torch.norm(c1 - c2, dim=1)
        sp_dists = sp_dists / (torch.max(sp_dists) + 1e-8)

        n_items = z_dists.size(0)
        # Revised spatial loss computation
        # spatial_loss = torch.sum((sp_dists - z_dists) ** 2) / n_items
        
        # spatial_loss = torch.sum(sp_dists * z_dists) / n_items
        
        #0.35 
        spatial_loss = torch.div(torch.sum(torch.mul(1.0 - z_dists, sp_dists)), n_items).to(self.device)



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

        self.model.eval()
        embeddings = []
        cell_ids = []
        self_attn_w1_list = []
        self_attn_w2_list = []
        neighbor_attn_w1_list = []
        neighbor_attn_w2_list = []
        neighbor_indices_list = []
        reduction_layer_list = []
    
        for batch in dataloader:
            batch = batch.to(self.device)
            with torch.no_grad():
                # Use the encoder directly to get embeddings
                pos_z = self.model.encoder(batch.x, batch.edge_index, batch.pos)
                embeddings.append(pos_z.cpu().numpy())
                cell_ids.extend(batch.cell_id)
    
                # Get attention weights
                (self_attn_w1, self_attn_w2, 
                 neighbor_attn_w1, neighbor_attn_w2, 
                 neighbor_indices, reduction_layer) = self.model.encoder.get_attention_info()
    
                self_attn_w1_list.append(self_attn_w1.cpu().numpy())
                self_attn_w2_list.append(self_attn_w2.cpu().numpy())
                neighbor_attn_w1_list.append(neighbor_attn_w1.cpu().numpy())
                neighbor_attn_w2_list.append(neighbor_attn_w2.cpu().numpy())

                reduction_layer_list.append(reduction_layer.cpu())
                neighbor_indices_list.extend(neighbor_indices)
    
        embeddings = np.concatenate(embeddings, axis=0)
        self_attn_w1 = np.concatenate(self_attn_w1_list, axis=0)
        self_attn_w2 = np.concatenate(self_attn_w2_list, axis=0)
    
        # Since neighbor attention weights and indices have variable lengths, store them as lists
        neighbor_attention = {
            'layer1': neighbor_attn_w1_list,
            'layer2': neighbor_attn_w2_list,
            'indices': neighbor_indices_list
        }
    
        # reduction_layes = np.concatenate(reduction_layer_list, axis=0)
        embedding_adata = anndata.AnnData(embeddings, obs=self.adata.obs.copy())
        embedding_adata.obsm['self_attention_weights_layer1'] = self_attn_w1
        embedding_adata.obsm['self_attention_weights_layer2'] = self_attn_w2
    
       
        embedding_adata.uns['neighbor_attention'] = neighbor_attention
        embedding_adata.uns['reduction_layes'] = reduction_layer_list
    
    
        self.embedding_adata = embedding_adata
    
        return embedding_adata
    
