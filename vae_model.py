import matplotlib
matplotlib.use('Agg')
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from pathlib import Path
import pickle
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
import base64
from PIL import Image
from io import BytesIO
from matplotlib.widgets import Slider
import matplotlib.colors as mcolors
from scipy.stats import pearsonr
from sklearn.metrics import silhouette_samples

class DigitDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels) if labels is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]
    

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim=72, encoding1_dim=64, encoding2_dim=32, latent_dim=10):
        super(VariationalAutoEncoder, self).__init__()
        
        # Encoder layers
        self.encoder_layers = nn.Sequential(
            nn.Linear(input_dim, encoding1_dim),
            nn.ReLU(),
            nn.Linear(encoding1_dim, encoding2_dim),
            nn.ReLU()
        )
        
        # Mean and log variance layers
        self.fc_mu = nn.Linear(encoding2_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoding2_dim, latent_dim)
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, encoding2_dim),
            nn.ReLU(),
            nn.Linear(encoding2_dim, encoding1_dim),
            nn.ReLU(),
            nn.Linear(encoding1_dim, input_dim),
            nn.Sigmoid()
        )
        
        self.latent_dim = latent_dim

    def encode(self, x):
        
        x = self.encoder_layers(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
    
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
    
        return self.decoder(z)
    
    def forward(self, x):
        
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        return z, decoded, mu, logvar
    
def vae_loss_function(reconstructed, x, mu, logvar, beta=1.0):
    
    # Reconstruction loss (binary cross entropy for binary data)
    # For MSE, use: recon_loss = F.mse_loss(reconstructed, x, reduction='sum')
    recon_loss = F.binary_cross_entropy(reconstructed, x, reduction='sum')
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    return recon_loss + beta * kl_div, recon_loss, kl_div


def save_models(vae, kmeans, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Save VAE
    torch.save(vae.state_dict(), f"{save_dir}/vae.pth")

    # Save KMeans
    with open(f"{save_dir}/kmeans.pkl", 'wb') as f:
        pickle.dump(kmeans, f)

    print(f"Models saved to {save_dir}")


def train_semi_supervised_vae(config):
    # Set seeds and device
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data with error handling
    print("Loading data...")
    try:
        train_data = pd.read_csv(config['train_data_path'])
        feature_columns = [f'pixel_{i}' for i in range(72)]
        
        # Check if all required columns exist
        missing_cols = [col for col in feature_columns if col not in train_data.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns in dataset: {missing_cols}")
            
        X_train = train_data[feature_columns].values
        y_train = train_data['label'].values if 'label' in train_data.columns else None
        
        # Validate data shape
        if X_train.shape[1] != config['input_dim']:
            raise ValueError(f"Input dimension mismatch: expected {config['input_dim']}, got {X_train.shape[1]}")
            
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

    # Create dataset and dataloader
    try:
        dataset = DigitDataset(X_train, y_train)
        data_loader = DataLoader(
            dataset, 
            batch_size=min(config['batch_size'], len(dataset)),
            shuffle=True
        )
    except Exception as e:
        print(f"Error creating dataloader: {str(e)}")
        raise

    # Initialize VAE model and optimizer
    print("Initializing VAE model...")
    try:
        vae = VariationalAutoEncoder(
            input_dim=config['input_dim'],
            encoding1_dim=config['encoding1_dim'],
            encoding2_dim=config['encoding2_dim'],
            latent_dim=config['latent_dim']
        ).to(device)

        optimizer = optim.Adam(vae.parameters(), lr=config['learning_rate'])
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        raise

    # Train VAE
    print("Training VAE...")
    vae.train()
    beta = config.get('beta', 1.0)
    use_label_guidance = config.get('use_label_guidance', True)
    contrastive_weight = config.get('contrastive_weight', 0.1)
    margin = config.get('margin', 1.0)
    
    try:
        for epoch in range(config['num_epochs']):
            running_loss = 0.0
            running_recon_loss = 0.0
            running_kl_loss = 0.0
            running_contrastive_loss = 0.0
            batch_count = 0
            
            for batch_features, batch_labels in data_loader:
                # Check for NaN or Inf in inputs
                if torch.isnan(batch_features).any() or torch.isinf(batch_features).any():
                    print(f"Warning: NaN or Inf found in batch features, skipping batch")
                    continue
                    
                batch_features = batch_features.to(device)
                if batch_labels is not None and use_label_guidance:
                    batch_labels = batch_labels.to(device)

                # Forward pass
                optimizer.zero_grad()
                try:
                    z, reconstructed, mu, logvar = vae(batch_features)
                except RuntimeError as e:
                    print(f"Forward pass error: {str(e)}")
                    print(f"Input shape: {batch_features.shape}")
                    continue

                # Check for NaN or Inf in outputs
                if torch.isnan(z).any() or torch.isinf(z).any() or \
                   torch.isnan(reconstructed).any() or torch.isinf(reconstructed).any():
                    print(f"Warning: NaN or Inf found in outputs, skipping batch")
                    continue

                # Compute VAE loss
                vae_loss, recon_loss, kl_loss = vae_loss_function(
                    reconstructed, batch_features, mu, logvar, beta=beta
                )
                
                # Initialize contrastive loss
                contrastive_loss = torch.tensor(0.0).to(device)

                # Clustering guidance loss 
                if batch_labels is not None and use_label_guidance:
                    try:
                        # Calculate pairwise distances in latent space
                        pairwise_dist = torch.cdist(z, z, p=2)
                        
                        # Create a mask for same-label pairs
                        label_matrix = batch_labels.unsqueeze(0) == batch_labels.unsqueeze(1)
                        
                        # Convert to float and to device
                        label_matrix = label_matrix.float()
                        
                        # Calculate contrastive loss
                        # For same label pairs: distance should be small
                        same_label_loss = label_matrix * pairwise_dist
                        
                        # For different label pairs: distance should be at least margin
                        diff_label_loss = (1 - label_matrix) * torch.clamp(margin - pairwise_dist, min=0)
                        
                        # Combine both components
                        contrastive_loss = (same_label_loss + diff_label_loss).mean()
                        
                        # Total loss with weighting
                        total_loss = vae_loss + contrastive_weight * contrastive_loss
                    except RuntimeError as e:
                        print(f"Contrastive loss error: {str(e)}")
                        total_loss = vae_loss 
                else:
                    total_loss = vae_loss

                # Check if loss is valid
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"Warning: Loss is NaN or Inf, skipping batch")
                    continue

                # Backward pass with error handling
                try:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
                    optimizer.step()
                except RuntimeError as e:
                    print(f"Backward pass error: {str(e)}")
                    continue

                running_loss += total_loss.item()
                running_recon_loss += recon_loss.item()
                running_kl_loss += kl_loss.item()
                running_contrastive_loss += contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else 0
                batch_count += 1

            # Print progress
            if batch_count > 0 and (epoch + 1) % 5 == 0:
                avg_loss = running_loss / batch_count
                avg_recon_loss = running_recon_loss / batch_count
                avg_kl_loss = running_kl_loss / batch_count
                avg_contrastive_loss = running_contrastive_loss / batch_count
                print(f'Epoch [{epoch + 1}/{config["num_epochs"]}], '
                      f'Total Loss: {avg_loss:.4f}, '
                      f'Recon Loss: {avg_recon_loss:.4f}, '
                      f'KL Loss: {avg_kl_loss:.4f}, '
                      f'Contrastive Loss: {avg_contrastive_loss:.4f}')
    except Exception as e:
        print(f"Training error: {str(e)}")
        print("Using partially trained model...")

   
    print("Extracting encodings...")
    vae.eval()
    all_encodings = []
    all_labels = []

    try:
        with torch.no_grad():
            for batch_features, batch_labels in data_loader:
                if torch.isnan(batch_features).any() or torch.isinf(batch_features).any():
                    continue
                    
                batch_features = batch_features.to(device)
                mu, _ = vae.encode(batch_features)
                
                # Check for valid encodings
                if torch.isnan(mu).any() or torch.isinf(mu).any():
                    continue
                    
                all_encodings.append(mu.cpu().numpy())
                if batch_labels is not None:
                    all_labels.append(batch_labels.numpy())

        if not all_encodings:
            raise ValueError("No valid encodings generated")
            
        encodings = np.concatenate(all_encodings)
        labels = np.concatenate(all_labels) if all_labels else None
    except Exception as e:
        print(f"Error extracting encodings: {str(e)}")
        raise

    
    print("Performing clustering...")
    clustering_method = config.get('clustering_method', 'kmeans')
    n_clusters = config.get('n_clusters', 10)
    cluster_labels = None

    try:
        if clustering_method == 'kmeans':
            # Standard KMeans
            kmeans = KMeans(
                n_clusters=n_clusters, 
                random_state=config['random_seed'],
                n_init=10  
            )
            cluster_labels = kmeans.fit_predict(encodings)
            clustering_model = kmeans

        elif clustering_method == 'gmm':
            # Gaussian Mixture Model
            from sklearn.mixture import GaussianMixture
            gmm = GaussianMixture(
                n_components=n_clusters,
                covariance_type='full',
                random_state=config['random_seed'],
                max_iter=100,
                n_init=5  
            )
            cluster_labels = gmm.fit_predict(encodings)
            clustering_model = gmm

        else:
            # Default to KMeans
            print(f"Unknown clustering method '{clustering_method}', using KMeans")
            kmeans = KMeans(n_clusters=config['n_clusters'], random_state=config['random_seed'], n_init='auto')
            cluster_labels = kmeans.fit_predict(encodings)
            clustering_model = kmeans
            
        # Validate that clustering worked    
        if cluster_labels is None or len(cluster_labels) != len(encodings):
            raise ValueError("Clustering failed to produce valid labels")
            
    except Exception as e:
        print(f"Clustering error: {str(e)}")
 
        print("Creating fallback clustering...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=config['random_seed'])
        cluster_labels = kmeans.fit_predict(encodings)
        clustering_model = kmeans

    # Save model
    try:
        save_models(vae, clustering_model, config['save_dir'])
    except Exception as e:
        print(f"Error saving models: {str(e)}")
        

    return vae, clustering_model, encodings, cluster_labels, labels, X_train


def create_vae_visualizations(vae, kmeans, encodings, X_train, y_train, config, cluster_labels, input_samples=None):

    try:
        import umap
        has_umap = True
    except ImportError:
        has_umap = False
        print("UMAP not available, skipping UMAP visualizations")

    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    # Create color maps for digits and clusters
    digit_cmap = plt.cm.get_cmap(name='tab10', lut=10)
    cluster_cmap = plt.cm.get_cmap(name='viridis', lut=config['n_clusters'])

    # 1. Cluster composition analysis (same as original)
    plt.figure(figsize=(16, 8))
    cluster_digit_composition = {}
    for cluster in range(config['n_clusters']):
        cluster_data = y_train[cluster_labels == cluster]
        unique, counts = np.unique(cluster_data, return_counts=True)
        cluster_digit_composition[cluster] = dict(zip(unique, counts))

    # Raw count version
    plt.subplot(121)
    bottom = np.zeros(len(range(config['n_clusters'])))

    for digit in range(10):
        digit_counts = [cluster_digit_composition[cluster].get(digit, 0)
                        for cluster in range(config['n_clusters'])]
        plt.bar(range(config['n_clusters']), digit_counts, bottom=bottom,
                label=f'Digit {digit}', color=digit_cmap(digit))
        bottom += digit_counts

    plt.title('Digit Composition in Each Cluster (Count)')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Samples')
    plt.xticks(range(config['n_clusters']))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Normalized version showing percentages
    plt.subplot(122)
    for cluster in range(config['n_clusters']):
        if cluster in cluster_digit_composition:
            total = sum(cluster_digit_composition[cluster].values())
            percentages = []
            for digit in range(10):
                if digit in cluster_digit_composition[cluster]:
                    percentages.append((digit, cluster_digit_composition[cluster][digit] / total * 100))

            percentages.sort(key=lambda x: x[1], reverse=True)

            bottom = 0
            for digit, percentage in percentages:
                plt.bar(cluster, percentage, bottom=bottom, color=digit_cmap(digit),
                        label=f'Digit {digit}' if cluster == 0 else "")

                # Add percentage text for significant contributions (>10%)
                if percentage > 10:
                    plt.text(cluster, bottom + percentage / 2, f'{digit}',
                             ha='center', va='center', color='white', fontweight='bold')

                bottom += percentage

    plt.title('Digit Composition in Each Cluster (%)')
    plt.xlabel('Cluster')
    plt.ylabel('Percentage')
    plt.xticks(range(config['n_clusters']))
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/cluster_composition.png", dpi=300)
    plt.close()

    # 2. Latent space visualization with multiple techniques
    # Prepare the latent vectors
    latent_vecs = encodings

    # Create a multi-panel visualization
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # 2.2 t-SNE visualization
    tsne = TSNE(n_components=2, random_state=config['random_seed'])
    latent_tsne = tsne.fit_transform(latent_vecs)

    ax3 = plt.subplot(gs[0, 0])
    scatter3 = ax3.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=y_train, cmap='tab10',
                           alpha=0.6, s=10)
    ax3.set_title('t-SNE - Digit Labels')
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')
    plt.colorbar(scatter3, ax=ax3, label='Digit')

    ax4 = plt.subplot(gs[1, 0])
    scatter4 = ax4.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=cluster_labels, cmap='viridis',
                           alpha=0.6, s=10)
    ax4.set_title('t-SNE - Cluster Labels')
    ax4.set_xlabel('t-SNE 1')
    ax4.set_ylabel('t-SNE 2')
    plt.colorbar(scatter4, ax=ax4, label='Cluster')

    # 2.3 UMAP visualization (if available)
    if has_umap:
        reducer = umap.UMAP(random_state=config['random_seed'])
        latent_umap = reducer.fit_transform(latent_vecs)

        ax5 = plt.subplot(gs[0, 1])
        scatter5 = ax5.scatter(latent_umap[:, 0], latent_umap[:, 1], c=y_train, cmap='tab10',
                               alpha=0.6, s=10)
        ax5.set_title('UMAP - Digit Labels')
        ax5.set_xlabel('UMAP 1')
        ax5.set_ylabel('UMAP 2')
        plt.colorbar(scatter5, ax=ax5, label='Digit')

        ax6 = plt.subplot(gs[1, 1])
        scatter6 = ax6.scatter(latent_umap[:, 0], latent_umap[:, 1], c=cluster_labels, cmap='viridis',
                               alpha=0.6, s=10)
        ax6.set_title('UMAP - Cluster Labels')
        ax6.set_xlabel('UMAP 1')
        ax6.set_ylabel('UMAP 2')
        plt.colorbar(scatter6, ax=ax6, label='Cluster')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/latent_space_visualizations.png", dpi=300)
    plt.close()

    # 3. VAE-specific latent space analysis - distribution of latent variables
    latent_dim = config['latent_dim']
    
    # 3.1 Create distribution plots for each latent dimension by digit class
    plt.figure(figsize=(20, 15))
    n_cols = 2
    n_rows = (latent_dim + n_cols - 1) // n_cols  # Ceiling division
    
    for dim in range(latent_dim):
        plt.subplot(n_rows, n_cols, dim + 1)
        for digit in range(10):
            digit_indices = y_train == digit
            if np.sum(digit_indices) > 0:  # Check if we have samples for this digit
                sns.kdeplot(encodings[digit_indices, dim], label=f'Digit {digit}')
        
        plt.title(f'Latent Dimension {dim} Distribution by Digit')
        plt.xlabel('Value')
        plt.ylabel('Density')
        if dim == 0:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
    plt.tight_layout()
    plt.savefig(f"{save_dir}/vae_latent_distributions.png", dpi=300)
    plt.close()
    
    # 3.2 Correlation heatmap (same as original)
    plt.figure(figsize=(15, 8))
    digit_onehot = np.eye(10)[y_train] 

    correlations = np.zeros((latent_dim, 10))
    for dim in range(latent_dim):
        for digit in range(10):
            correlations[dim, digit] = pearsonr(latent_vecs[:, dim], digit_onehot[:, digit])[0]

    sns.heatmap(correlations, annot=True, cmap='coolwarm',
            xticklabels=range(10), yticklabels=[f'Dim {i}' for i in range(latent_dim)])
    plt.title('Correlation: Latent Dimensions vs Digit Classes')
    plt.xlabel('Digit')
    plt.ylabel('Latent Dimension')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/latent_dimension_correlation.png", dpi=300)
    plt.close()

    # 4. Input-to-output visualization with samples
    # If custom inputs are provided, use them
    if input_samples is not None:
        X_samples = input_samples
        # If labels are provided with custom samples (should be a tuple)
        if isinstance(input_samples, tuple) and len(input_samples) == 2:
            X_samples, sample_labels = input_samples
        else:
            # Generate placeholder labels
            sample_labels = np.zeros(len(X_samples))
    else:
        # Sample digits from the training set
        sample_indices = []
        for digit in range(10):
            digit_indices = np.where(y_train == digit)[0]
            if len(digit_indices) >= 3:
                sample_indices.extend(np.random.choice(digit_indices, 3, replace=False))

        X_samples = X_train[sample_indices]
        sample_labels = y_train[sample_indices]

    # Encode and decode the samples
    vae.eval()
    X_tensor = torch.FloatTensor(X_samples)

    with torch.no_grad():
        mu, logvar = vae.encode(X_tensor)
        z = vae.reparameterize(mu, logvar)
        reconstructed = vae.decode(z)

    mu = mu.numpy()
    logvar = logvar.numpy()
    z = z.numpy()
    reconstructed = reconstructed.numpy()

    # Create a figure showing original, encoding (mean/variance), and reconstruction
    fig = plt.figure(figsize=(20, len(X_samples)))
    gs = gridspec.GridSpec(len(X_samples), 4)

    for i in range(len(X_samples)):
        # Original
        ax1 = plt.subplot(gs[i, 0])
        ax1.imshow(X_samples[i].reshape(12, 6), cmap='gray_r')
        if i == 0:
            ax1.set_title('Original Input')
        if hasattr(sample_labels[i], 'item'):  # Handle tensor labels
            digit = sample_labels[i].item()
        else:
            digit = sample_labels[i]
        ax1.set_ylabel(f'Sample {i} (Digit {int(digit)})')
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Mean of latent encoding (bar chart)
        ax2 = plt.subplot(gs[i, 1])
        ax2.bar(range(latent_dim), mu[i], color='steelblue')
        if i == 0:
            ax2.set_title('Latent Space Mean (μ)')
        ax2.set_xticks(range(latent_dim))
        ax2.set_xticklabels([f'D{j}' for j in range(latent_dim)], rotation=45)

        # Log variance of latent encoding (bar chart)
        ax3 = plt.subplot(gs[i, 2])
        ax3.bar(range(latent_dim), logvar[i], color='salmon')
        if i == 0:
            ax3.set_title('Latent Space LogVar (log σ²)')
        ax3.set_xticks(range(latent_dim))
        ax3.set_xticklabels([f'D{j}' for j in range(latent_dim)], rotation=45)

        # Reconstruction
        ax4 = plt.subplot(gs[i, 3])
        ax4.imshow(reconstructed[i].reshape(12, 6), cmap='gray_r')
        if i == 0:
            ax4.set_title('Reconstruction')
        ax4.set_xticks([])
        ax4.set_yticks([])

    plt.tight_layout()
    plt.savefig(f"{save_dir}/vae_input_output_visualization.png", dpi=300)
    plt.close()

    # 5. Latent space traversal - similar to original but using the decoder
    if config['latent_dim'] >= 2:
        # Create latent space traversals for each dimension
        plt.figure(figsize=(20, latent_dim * 3))
        gs = gridspec.GridSpec(latent_dim, 10)

        # Create a reference latent vector (average encoding)
        reference_latent = np.mean(encodings, axis=0)

        for dim in range(latent_dim):
            # Get min and max values for this dimension
            dim_values = latent_vecs[:, dim]
            dim_min, dim_max = np.percentile(dim_values, [5, 95])

            # Create a range of values for this dimension
            traversal_values = np.linspace(dim_min, dim_max, 10)

            # For each value, generate an image
            for i, val in enumerate(traversal_values):
                latent = reference_latent.copy()
                latent[dim] = val

                with torch.no_grad():
                    generated = vae.decode(torch.FloatTensor(latent).unsqueeze(0))
                    img = generated.squeeze().numpy().reshape(12, 6)

                # Plot
                ax = plt.subplot(gs[dim, i])
                ax.imshow(img, cmap='gray_r')
                if i == 0:
                    ax.set_ylabel(f'Dim {dim}')
                if dim == 0:
                    ax.set_title(f'{val:.2f}')
                ax.set_xticks([])
                ax.set_yticks([])

        plt.suptitle('Effect of Each Latent Dimension (Columns: Min to Max Value)', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(f"{save_dir}/latent_dimension_traversal.png", dpi=300)
        plt.close()

     # 6. VAE Random Sampling - Generate random samples from latent space
    n_samples_per_row = 5
    n_rows = 4
    total_samples = n_samples_per_row * n_rows
    
    plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(n_rows, n_samples_per_row)
    
    # Sample random points from a normal distribution
    random_latent = np.random.normal(0, 1, size=(total_samples, config['latent_dim']))
    
    # Decode the random latent vectors
    with torch.no_grad():
        random_gen = vae.decode(torch.FloatTensor(random_latent))
        random_imgs = random_gen.numpy()
    
    # Plot the generated images
    for i in range(total_samples):
        row = i // n_samples_per_row
        col = i % n_samples_per_row
        
        ax = plt.subplot(gs[row, col])
        ax.imshow(random_imgs[i].reshape(12, 6), cmap='gray_r')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle('Random Samples from VAE Latent Space', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{save_dir}/vae_random_samples.png", dpi=300)
    plt.close()
    
    # 7. Cluster silhouette analysis
    plt.figure(figsize=(12, 8))
    
    # Calculate silhouette scores for each sample
    silhouette_vals = silhouette_samples(encodings, cluster_labels)
    
    # Organize by cluster
    y_ticks = []
    y_lower, y_upper = 0, 0
    
    for i in range(config['n_clusters']):
        cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
        cluster_silhouette_vals.sort()
        
        y_upper += len(cluster_silhouette_vals)
        
        plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, 
                 height=1.0, edgecolor='none', color=cluster_cmap(i / config['n_clusters']))
        
        y_ticks.append((y_lower + y_upper) / 2)
        y_lower += len(cluster_silhouette_vals)
    
    # Add the average silhouette score line
    avg_silhouette = np.mean(silhouette_vals)
    plt.axvline(avg_silhouette, color="red", linestyle="--", 
                label=f'Average: {avg_silhouette:.3f}')
    
    plt.yticks(y_ticks, [f'Cluster {i}' for i in range(config['n_clusters'])])
    plt.xlabel('Silhouette Coefficient')
    plt.ylabel('Cluster')
    plt.title('Silhouette Analysis for Clustering')
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/cluster_silhouette.png", dpi=300)
    plt.close()
    
    print(f"All visualizations saved to {save_dir}")
    return cluster_digit_composition


def run_vae_clustering_pipeline(config):

    # Set up directories
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Train the VAE and perform clustering
    vae, clustering_model, encodings, cluster_labels, labels, X_train = train_semi_supervised_vae(config)
    
    # Generate visualizations
    create_vae_visualizations(vae, clustering_model, encodings, X_train, labels, config, cluster_labels)
    
    return vae, clustering_model, encodings, cluster_labels


# Example configuration
if __name__ == "__main__":
    config = {
        'train_data_path': 'final_datasets/train_dataset.csv',  
        'input_dim': 72,              
        'encoding1_dim': 64,          
        'encoding2_dim': 32,          
        'latent_dim': 10,             
        'n_clusters': 10,           
        'learning_rate': 0.001,     
        'batch_size': 128,           
        'num_epochs': 500,           
        'random_seed': 42,           
        'save_dir': 'models/vae_kmeans_clustering', 
        'beta': 1.0,                  
        'use_label_guidance': True, 
        'contrastive_weight': 0.1,    
        'margin': 1.0,                
        'clustering_method': 'kmeans' # kmeans , gmm
    }
    
    
    vae, clustering_model, encodings, cluster_labels = run_vae_clustering_pipeline(config)
    
    print("VAE clustering pipeline completed successfully!")
