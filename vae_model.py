import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import base64
from PIL import Image
from io import BytesIO
import umap
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import os


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
    def __init__(self, input_dim=72, encoding_dim=32, latent_dim=10):
        super(VariationalAutoEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.BatchNorm1d(encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, encoding_dim // 2),
            nn.BatchNorm1d(encoding_dim // 2),
            nn.ReLU(),
        )

        # Mean and variance for the latent distribution
        self.fc_mu = nn.Linear(encoding_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(encoding_dim // 2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, encoding_dim // 2),
            nn.BatchNorm1d(encoding_dim // 2),
            nn.ReLU(),
            nn.Linear(encoding_dim // 2, encoding_dim),
            nn.BatchNorm1d(encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

        self.latent_dim = latent_dim

    def encode(self, x):
        # Get encoded representation
        encoded = self.encoder(x)
        # Get mean and log variance
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        # Decode from latent space
        return self.decoder(z)

    def forward(self, x):
        # Encode, sample, and decode
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        return mu, logvar, z, decoded

    def sample(self, num_samples, device='cpu'):
        # Generate random samples from the latent space
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    # Reconstruction loss (binary cross entropy for binary data)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss with beta weighting for KL term (beta-VAE)
    return BCE + beta * KLD, BCE, KLD


def save_models(vae, kmeans, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Save VAE
    torch.save(vae.state_dict(), f"{save_dir}/vae.pth")

    # Save KMeans
    with open(f"{save_dir}/kmeans.pkl", 'wb') as f:
        pickle.dump(kmeans, f)

    print(f"Models saved to {save_dir}")


def create_detailed_visualizations(vae, kmeans, encodings, X_train, y_train, config):
    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    # Extract useful data
    cluster_labels = kmeans.labels_

    # Create color maps for digits and clusters
    digit_cmap = plt.cm.get_cmap('tab10', 10)
    cluster_cmap = plt.cm.get_cmap('viridis', config['n_clusters'])

    # 1. Cluster composition analysis - Create a more detailed stacked bar chart
    plt.figure(figsize=(16, 8))
    cluster_digit_composition = {}
    for cluster in range(config['n_clusters']):
        cluster_data = y_train[cluster_labels == cluster]
        unique, counts = np.unique(cluster_data, return_counts=True)
        cluster_digit_composition[cluster] = dict(zip(unique, counts))

    # Prepare data for stacked bar plot
    cluster_names = list(range(config['n_clusters']))

    # Create a normalized version showing percentages
    plt.subplot(121)
    bottom = np.zeros(len(cluster_names))

    for digit in range(10):
        digit_counts = [cluster_digit_composition[cluster].get(digit, 0) for cluster in cluster_names]
        plt.bar(cluster_names, digit_counts, bottom=bottom, label=f'Digit {digit}', color=digit_cmap(digit))
        bottom += digit_counts

    plt.title('Digit Composition in Each Cluster (Count)')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Samples')
    plt.xticks(cluster_names)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Normalized version
    plt.subplot(122)
    for cluster in cluster_names:
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
    plt.xticks(cluster_names)
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/cluster_composition.png", dpi=300)
    plt.close()

    # 2. Latent space visualization with multiple dimensionality reduction techniques
    # Prepare the latent vectors
    latent_vecs = encodings

    # Create a multi-panel visualization
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(2, 3)

    # 2.1 PCA visualization
    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latent_vecs)

    ax1 = plt.subplot(gs[0, 0])
    scatter1 = ax1.scatter(latent_pca[:, 0], latent_pca[:, 1], c=y_train, cmap='tab10',
                           alpha=0.6, s=10)
    ax1.set_title(f'PCA - Digit Labels (Explained var: {pca.explained_variance_ratio_.sum():.2f})')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    plt.colorbar(scatter1, ax=ax1, label='Digit')

    ax2 = plt.subplot(gs[0, 1])
    scatter2 = ax2.scatter(latent_pca[:, 0], latent_pca[:, 1], c=cluster_labels, cmap='viridis',
                           alpha=0.6, s=10)
    ax2.set_title('PCA - Cluster Labels')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    plt.colorbar(scatter2, ax=ax2, label='Cluster')

    # 2.2 t-SNE visualization
    tsne = TSNE(n_components=2, random_state=config['random_seed'])
    latent_tsne = tsne.fit_transform(latent_vecs)

    ax3 = plt.subplot(gs[0, 2])
    scatter3 = ax3.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=y_train, cmap='tab10',
                           alpha=0.6, s=10)
    ax3.set_title('t-SNE - Digit Labels')
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')

    ax4 = plt.subplot(gs[1, 0])
    scatter4 = ax4.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=cluster_labels, cmap='viridis',
                           alpha=0.6, s=10)
    ax4.set_title('t-SNE - Cluster Labels')
    ax4.set_xlabel('t-SNE 1')
    ax4.set_ylabel('t-SNE 2')
    plt.colorbar(scatter4, ax=ax4, label='Cluster')

    # 2.3 UMAP visualization
    reducer = umap.UMAP(random_state=config['random_seed'])
    latent_umap = reducer.fit_transform(latent_vecs)

    ax5 = plt.subplot(gs[1, 1])
    scatter5 = ax5.scatter(latent_umap[:, 0], latent_umap[:, 1], c=y_train, cmap='tab10',
                           alpha=0.6, s=10)
    ax5.set_title('UMAP - Digit Labels')
    ax5.set_xlabel('UMAP 1')
    ax5.set_ylabel('UMAP 2')
    plt.colorbar(scatter5, ax=ax5, label='Digit')

    ax6 = plt.subplot(gs[1, 2])
    scatter6 = ax6.scatter(latent_umap[:, 0], latent_umap[:, 1], c=cluster_labels, cmap='viridis',
                           alpha=0.6, s=10)
    ax6.set_title('UMAP - Cluster Labels')
    ax6.set_xlabel('UMAP 1')
    ax6.set_ylabel('UMAP 2')
    plt.colorbar(scatter6, ax=ax6, label='Cluster')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/latent_space_visualizations.png", dpi=300)
    plt.close()

    # 3. Reconstruction quality visualization
    # Sample a few digits from each class for reconstruction visualization
    plt.figure(figsize=(20, 12))
    vae.eval()

    n_rows = 10  # One row per digit
    n_cols = 6  # Original and 5 reconstructions

    sample_indices = []
    for digit in range(10):
        digit_indices = np.where(y_train == digit)[0]
        if len(digit_indices) >= 5:
            sample_indices.extend(np.random.choice(digit_indices, 5, replace=False))

    X_samples = X_train[sample_indices]
    y_samples = y_train[sample_indices]

    # Convert to tensor for reconstruction
    X_tensor = torch.FloatTensor(X_samples)

    # Get reconstructions
    with torch.no_grad():
        _, _, _, reconstructed = vae(X_tensor)

    reconstructed = reconstructed.numpy()

    # Plot original and reconstructed digits
    for i, digit in enumerate(range(10)):
        digit_indices = [j for j, y in enumerate(y_samples) if y == digit]
        if not digit_indices:
            continue

        for j, idx in enumerate(digit_indices[:5]):
            # Original
            plt.subplot(n_rows, n_cols, i * n_cols + j + 1)
            # For 8x9 digits
            pixels = X_samples[idx].reshape(8, 9)
            plt.imshow(pixels, cmap='gray')
            plt.title(f"Original {digit}")
            plt.axis('off')

            # Reconstructed
            plt.subplot(n_rows, n_cols, i * n_cols + j + 1 + 5)
            pixels_recon = reconstructed[idx].reshape(8, 9)
            plt.imshow(pixels_recon, cmap='gray')
            plt.title(f"Reconstructed {digit}")
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/reconstruction_examples.png", dpi=300)
    plt.close()

    # 4. Latent space interpolation
    plt.figure(figsize=(15, 10))

    # Choose two digits to interpolate between
    digit_pairs = [(0, 1), (3, 8), (4, 9), (6, 8)]
    n_steps = 10

    vae.eval()

    for pair_idx, (digit1, digit2) in enumerate(digit_pairs):
        # Get sample indices for the digit pair
        digit1_indices = np.where(y_train == digit1)[0]
        digit2_indices = np.where(y_train == digit2)[0]

        # Take first sample of each digit
        sample1_idx = np.random.choice(digit1_indices, 1)[0]
        sample2_idx = np.random.choice(digit2_indices, 1)[0]

        sample1 = torch.FloatTensor(X_train[sample1_idx]).unsqueeze(0)
        sample2 = torch.FloatTensor(X_train[sample2_idx]).unsqueeze(0)

        # Encode both samples to get latent vectors
        with torch.no_grad():
            mu1, _ = vae.encode(sample1)
            mu2, _ = vae.encode(sample2)

        # Create interpolations in latent space
        alphas = np.linspace(0, 1, n_steps)
        interpolations = []

        for alpha in alphas:
            # Interpolate between the means
            z_interp = alpha * mu2 + (1 - alpha) * mu1

            # Decode the interpolated latent vector
            with torch.no_grad():
                decoded = vae.decode(z_interp)

            interpolations.append(decoded.squeeze().numpy())

        # Plot the interpolations
        for i, interp in enumerate(interpolations):
            plt.subplot(len(digit_pairs), n_steps, pair_idx * n_steps + i + 1)
            plt.imshow(interp.reshape(8, 9), cmap='gray')
            if i == 0:
                plt.title(f"{digit1} → {digit2}   {alphas[i]:.1f}", fontsize=10)
            else:
                plt.title(f"{alphas[i]:.1f}", fontsize=10)
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/latent_interpolations.png", dpi=300)
    plt.close()

    # 5. Latent space generation
    plt.figure(figsize=(12, 12))

    # Generate samples from the latent space
    n_samples = 100  # 10x10 grid

    # Create a grid of values in a 2D subspace of the latent space
    if config['latent_dim'] >= 2:
        # Use the first two dimensions
        grid_size = int(np.sqrt(n_samples))
        x = np.linspace(-3, 3, grid_size)
        y = np.linspace(-3, 3, grid_size)

        xv, yv = np.meshgrid(x, y)
        z_grid = np.zeros((n_samples, config['latent_dim']))

        # Set the first two dimensions based on the grid
        z_grid[:, 0] = xv.flatten()
        z_grid[:, 1] = yv.flatten()

        # Convert to tensor
        z_tensor = torch.FloatTensor(z_grid)

        # Decode the latent vectors
        with torch.no_grad():
            generated = vae.decode(z_tensor)

        generated = generated.numpy()

        # Plot the generated digits
        for i in range(n_samples):
            plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(generated[i].reshape(8, 9), cmap='gray')
            plt.axis('off')

        plt.suptitle("Generated Digits from 2D Latent Space Grid", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/latent_space_generation.png", dpi=300)
        plt.close()

    # 6. Interactive 3D visualization
    # Perform t-SNE to 3D
    tsne_3d = TSNE(n_components=3, random_state=config['random_seed'])
    latent_tsne_3d = tsne_3d.fit_transform(latent_vecs)

    # Create dual visualizations - one by digit, one by cluster
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=('Colored by Digit', 'Colored by Cluster')
    )

    # First plot - colored by digit
    fig.add_trace(
        go.Scatter3d(
            x=latent_tsne_3d[:, 0],
            y=latent_tsne_3d[:, 1],
            z=latent_tsne_3d[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=y_train,
                colorscale='Turbo',
                opacity=0.8,
                colorbar=dict(
                    title="Digit",
                    x=0.45
                )
            ),
            text=[f"Digit: {y}<br>Cluster: {c}" for y, c in zip(y_train, cluster_labels)],
            hoverinfo='text'
        ),
        row=1, col=1
    )

    # Second plot - colored by cluster
    fig.add_trace(
        go.Scatter3d(
            x=latent_tsne_3d[:, 0],
            y=latent_tsne_3d[:, 1],
            z=latent_tsne_3d[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=cluster_labels,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(
                    title="Cluster",
                    x=1.0
                )
            ),
            text=[f"Digit: {y}<br>Cluster: {c}" for y, c in zip(y_train, cluster_labels)],
            hoverinfo='text'
        ),
        row=1, col=2
    )

    fig.update_layout(
        title='3D Visualization of VAE Latent Space',
        height=800,
        scene=dict(
            xaxis_title='t-SNE 1',
            yaxis_title='t-SNE 2',
            zaxis_title='t-SNE 3'
        ),
        scene2=dict(
            xaxis_title='t-SNE 1',
            yaxis_title='t-SNE 2',
            zaxis_title='t-SNE 3'
        )
    )

    # Save interactive plot
    fig.write_html(f"{save_dir}/3d_latent_visualization.html")

    # 7. Create a beta-VAE disentanglement visualization if latent dim >= 2
    if config['latent_dim'] >= 2:
        # Choose two latent dimensions for visualization
        dim1, dim2 = 0, 1

        plt.figure(figsize=(15, 15))
        grid_size = 8
        z = torch.zeros(grid_size * grid_size, config['latent_dim'])

        # Values to traverse
        values = np.linspace(-3, 3, grid_size)

        # Assign values to the chosen dimensions
        for i, v1 in enumerate(values):
            for j, v2 in enumerate(values):
                idx = i * grid_size + j
                z[idx, dim1] = v1
                z[idx, dim2] = v2

        # Decode the traversed latent vectors
        with torch.no_grad():
            generated = vae.decode(z)

        # Plot the traversals
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                plt.subplot(grid_size, grid_size, idx + 1)
                plt.imshow(generated[idx].numpy().reshape(8, 9), cmap='gray')
                plt.axis('off')

        plt.tight_layout()
        plt.suptitle(f"Latent Space Traversal - Dimensions {dim1} and {dim2}", fontsize=16)
        plt.savefig(f"{save_dir}/latent_traversal.png", dpi=300)
        plt.close()

    # 8. Create a cluster centroid visualization
    plt.figure(figsize=(12, 4))
    # Get cluster centroids
    centroids = kmeans.cluster_centers_

    # Decode the centroids
    with torch.no_grad():
        centroid_decoded = vae.decode(torch.FloatTensor(centroids))

    # Plot the centroid reconstructions
    for i in range(config['n_clusters']):
        plt.subplot(2, config['n_clusters'] // 2, i + 1)
        plt.imshow(centroid_decoded[i].numpy().reshape(8, 9), cmap='gray')
        plt.title(f"Cluster {i}")
        plt.axis('off')

    plt.tight_layout()
    plt.suptitle("Cluster Centroids", fontsize=16)
    plt.savefig(f"{save_dir}/cluster_centroids.png", dpi=300)
    plt.close()

    print("Detailed Visualization Completed:")
    print(f"- Cluster Composition: {save_dir}/cluster_composition.png")
    print(f"- Latent Space Visualizations: {save_dir}/latent_space_visualizations.png")
    print(f"- Reconstruction Examples: {save_dir}/reconstruction_examples.png")
    print(f"- Latent Space Interpolations: {save_dir}/latent_interpolations.png")
    print(f"- Latent Space Generation: {save_dir}/latent_space_generation.png")
    print(f"- Interactive 3D Visualization: {save_dir}/3d_latent_visualization.html")
    print(f"- Latent Space Traversal: {save_dir}/latent_traversal.png")
    print(f"- Cluster Centroids: {save_dir}/cluster_centroids.png")


def train_and_save_vae_model(config):
    # Set seeds and device
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    try:
        train_data = pd.read_csv(config['train_data_path'])
        feature_columns = [f'pixel_{i}' for i in range(config['input_dim'])]
        X_train = train_data[feature_columns].values
        y_train = train_data['label'].values if 'label' in train_data.columns else None
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Create dataset and dataloader
    dataset = DigitDataset(X_train, y_train)
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # Initialize model and optimizer
    print("Initializing VAE model...")
    vae = VariationalAutoEncoder(
        input_dim=config['input_dim'],
        encoding_dim=config['encoding_dim'],
        latent_dim=config['latent_dim']
    ).to(device)

    optimizer = optim.Adam(vae.parameters(), lr=config['learning_rate'])

    # Initialize tracking variables for losses
    train_losses = []
    reconstruction_losses = []
    kl_losses = []

    # Train VAE
    print("Training VAE...")
    vae.train()
    for epoch in range(config['num_epochs']):
        running_loss = 0.0
        running_bce = 0.0
        running_kld = 0.0

        for batch_features, _ in data_loader:
            batch_features = batch_features.to(device)

            # Forward pass
            optimizer.zero_grad()
            mu, logvar, _, reconstructed = vae(batch_features)

            # Calculate loss with adaptive beta
            beta = min(1.0, epoch / (0.75 * config['num_epochs']))  # Gradually increase beta
            loss, bce, kld = vae_loss_function(reconstructed, batch_features, mu, logvar, beta=beta)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_bce += bce.item()
            running_kld += kld.item()

        # Print progress
        if (epoch + 1) % 5 == 0:
            avg_loss = running_loss / len(data_loader)
            avg_bce = running_bce / len(data_loader)
            avg_kld = running_kld / len(data_loader)
            train_losses.append(avg_loss)
            reconstruction_losses.append(avg_bce)
            kl_losses.append(avg_kld)
            print(f'Epoch [{epoch + 1}/{config["num_epochs"]}], ' +
                  f'Loss: {avg_loss:.4f}, ' +
                  f'BCE: {avg_bce:.4f}, ' +
                  f'KLD: {avg_kld:.4f}, ' +
                  f'Beta: {beta:.2f}')

    # Plot training loss
    if config['visualize']:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(0, config['num_epochs'], 5), train_losses, label='Total Loss')
        plt.plot(range(0, config['num_epochs'], 5), reconstruction_losses, label='Reconstruction Loss')
        plt.plot(range(0, config['num_epochs'], 5), kl_losses, label='KL Divergence')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Components')
        plt.legend()

        plt.subplot(1, 2, 2)
        betas = [min(1.0, epoch / (0.75 * config['num_epochs'])) for epoch in range(0, config['num_epochs'], 5)]
        plt.plot(range(0, config['num_epochs'], 5), betas)
        plt.xlabel('Epoch')
        plt.ylabel('Beta Value')
        plt.title('Beta Schedule')

        plt.tight_layout()
        plt.savefig(f"{config['save_dir']}/training_loss.png")
        plt.close()

    # Get latent representations for clustering
    print("Getting latent representations for clustering...")
    vae.eval()
    latent_vecs = []

    with torch.no_grad():
        for batch_features, _ in data_loader:
            batch_features = batch_features.to(device)
            mu, _, _, _ = vae(batch_features)
            latent_vecs.append(mu.cpu().numpy())

    latent_vecs = np.concatenate(latent_vecs)

    # Perform clustering
    print("Performing clustering...")
    kmeans = KMeans(n_clusters=config['n_clusters'], random_state=config['random_seed'], n_init=10)
    cluster_labels = kmeans.fit_predict(latent_vecs)

    # Save models
    print("Saving models...")
    save_models(vae, kmeans, config['save_dir'])

    # Visualize results
    if config['visualize']:
        print("Generating visualizations...")
        create_detailed_visualizations(
            vae,
            kmeans,
            latent_vecs,
            X_train,
            y_train,
            config
        )

    return vae, kmeans, cluster_labels


def main():
    # Configuration with expanded options
    config = {
        'random_seed': 42,
        'batch_size': 512,  # Smaller batch size for better stochasticity
        'num_epochs': 200,  # Can be lower due to more efficient architecture
        'learning_rate': 0.001,
        'input_dim': 72,
        'encoding_dim': 32,  # Higher dimensional encoding for better feature extraction
        'latent_dim': 10,
        'n_clusters': 10,
        'train_data_path': 'final_datasets/train_dataset.csv',
        'save_dir': './models/vae_clustering',
        'visualize': True,
        'beta_schedule': 'linear',  # Options: 'constant', 'linear', 'cyclical'
        'early_stopping': True,
        'patience': 20,
        'min_delta': 0.001
    }

    # Train and save model
    vae, kmeans, cluster_labels = train_and_save_vae_model(config)

    print("\nTraining and saving completed!")
    print(f"Models saved in {config['save_dir']}")
    if config['visualize']:
        print(f"Visualizations saved in {config['save_dir']}")

    # Generate additional interactive visualization
    if config['visualize']:
        create_interactive_dashboard(vae, kmeans, config)


def create_interactive_dashboard(vae, kmeans, config):
    """Create an interactive HTML dashboard with Plotly"""
    try:
        # This would be done after training with the latent representations
        print("Creating interactive dashboard...")

        # Load data to visualize
        train_data = pd.read_csv(config['train_data_path'])
        feature_columns = [f'pixel_{i}' for i in range(config['input_dim'])]
        X_train = train_data[feature_columns].values
        y_train = train_data['label'].values if 'label' in train_data.columns else None

        # Get latent representations
        vae.eval()
        X_tensor = torch.FloatTensor(X_train)
        with torch.no_grad():
            mu, logvar, z, reconstructed = vae(X_tensor)

        latent_vecs = z.numpy()
        cluster_labels = kmeans.predict(mu.numpy())

        # Perform dimensionality reduction for visualization
        pca = PCA(n_components=3)
        tsne = TSNE(n_components=3, random_state=config['random_seed'])
        umap_reducer = umap.UMAP(n_components=3, random_state=config['random_seed'])

        # Calculate all three projections
        pca_result = pca.fit_transform(latent_vecs)
        tsne_result = tsne.fit_transform(latent_vecs)
        umap_result = umap_reducer.fit_transform(latent_vecs)

        # Calculate reconstruction error for each sample
        reconstruction_error = np.mean(np.square(X_train - reconstructed.numpy()), axis=1)

        # Create a dataframe for easier plotting
        viz_df = pd.DataFrame({
            'digit': y_train,
            'cluster': cluster_labels,
            'recon_error': reconstruction_error,
            'pca_1': pca_result[:, 0],
            'pca_2': pca_result[:, 1],
            'pca_3': pca_result[:, 2],
            'tsne_1': tsne_result[:, 0],
            'tsne_2': tsne_result[:, 1],
            'tsne_3': tsne_result[:, 2],
            'umap_1': umap_result[:, 0],
            'umap_2': umap_result[:, 1],
            'umap_3': umap_result[:, 2]
        })

        # Create the interactive dashboard
        dash_html = create_plotly_dashboard(viz_df, config)

        # Save the dashboard
        with open(f"{config['save_dir']}/interactive_dashboard.html", 'w') as f:
            f.write(dash_html)

        print(f"Interactive dashboard saved to {config['save_dir']}/interactive_dashboard.html")

    except Exception as e:
        print(f"Error creating dashboard: {e}")


def create_plotly_dashboard(df, config):
    """Create a combined HTML dashboard using Plotly"""

    # Create the layout for the dashboard
    dashboard = make_subplots(
        rows=2, cols=3,
        specs=[
            [{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}],
            [{'colspan': 3, 'type': 'bar'}, None, None]
        ],
        subplot_titles=(
            'PCA Projection', 't-SNE Projection', 'UMAP Projection',
            'Cluster Analysis'
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )

    # Color scales
    digit_colorscale = 'Turbo'  # Good for categorical data like digits
    cluster_colorscale = 'Viridis'  # Good for clusters

    # Add traces for PCA projection
    dashboard.add_trace(
        go.Scatter3d(
            x=df['pca_1'],
            y=df['pca_2'],
            z=df['pca_3'],
            mode='markers',
            marker=dict(
                size=4,
                color=df['digit'],
                colorscale=digit_colorscale,
                opacity=0.7,
                showscale=True,
                colorbar=dict(
                    title="Digit",
                    x=0.3,
                    len=0.4
                )
            ),
            text=[f"Digit: {d}<br>Cluster: {c}<br>Error: {e:.4f}"
                  for d, c, e in zip(df['digit'], df['cluster'], df['recon_error'])],
            hoverinfo='text',
            name='Color by Digit'
        ),
        row=1, col=1
    )

    # Add traces for t-SNE projection
    dashboard.add_trace(
        go.Scatter3d(
            x=df['tsne_1'],
            y=df['tsne_2'],
            z=df['tsne_3'],
            mode='markers',
            marker=dict(
                size=4,
                color=df['cluster'],
                colorscale=cluster_colorscale,
                opacity=0.7,
                showscale=True,
                colorbar=dict(
                    title="Cluster",
                    x=0.65,
                    len=0.4
                )
            ),
            text=[f"Digit: {d}<br>Cluster: {c}<br>Error: {e:.4f}"
                  for d, c, e in zip(df['digit'], df['cluster'], df['recon_error'])],
            hoverinfo='text',
            name='Color by Cluster'
        ),
        row=1, col=2
    )

    # Add traces for UMAP projection
    dashboard.add_trace(
        go.Scatter3d(
            x=df['umap_1'],
            y=df['umap_2'],
            z=df['umap_3'],
            mode='markers',
            marker=dict(
                size=4,
                color=df['recon_error'],
                colorscale='Plasma',
                opacity=0.7,
                showscale=True,
                colorbar=dict(
                    title="Error",
                    x=1.0,
                    len=0.4
                )
            ),
            text=[f"Digit: {d}<br>Cluster: {c}<br>Error: {e:.4f}"
                  for d, c, e in zip(df['digit'], df['cluster'], df['recon_error'])],
            hoverinfo='text',
            name='Color by Error'
        ),
        row=1, col=3
    )

    # Create cluster analysis - how many of each digit in each cluster
    cluster_digit_counts = {}
    for cluster in range(config['n_clusters']):
        cluster_data = df[df['cluster'] == cluster]
        if 'digit' in cluster_data.columns:
            counts = cluster_data['digit'].value_counts().sort_index()
            cluster_digit_counts[cluster] = counts

    # Prepare data for stacked bar chart
    clusters = []
    digits = []
    values = []
    for cluster in range(config['n_clusters']):
        if cluster in cluster_digit_counts:
            for digit, count in cluster_digit_counts[cluster].items():
                clusters.append(f'Cluster {cluster}')
                digits.append(f'Digit {digit}')
                values.append(count)

    # Add stacked bar chart
    dashboard.add_trace(
        go.Bar(
            x=clusters,
            y=values,
            text=digits,
            hoverinfo='text+y',
            marker=dict(
                color=[int(d.split()[-1]) for d in digits],
                colorscale=digit_colorscale
            ),
            name='Digit Distribution'
        ),
        row=2, col=1
    )

    # Update layout
    dashboard.update_layout(
        title='VAE Clustering Interactive Dashboard',
        height=900,
        width=1200,
        barmode='stack',
        scene=dict(
            xaxis_title='PCA 1',
            yaxis_title='PCA 2',
            zaxis_title='PCA 3'
        ),
        scene2=dict(
            xaxis_title='t-SNE 1',
            yaxis_title='t-SNE 2',
            zaxis_title='t-SNE 3'
        ),
        scene3=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3'
        ),
        xaxis=dict(title='Cluster'),
        yaxis=dict(title='Count'),
        showlegend=False
    )

    # Create HTML
    return dashboard.to_html(include_plotlyjs=True, full_html=True)


def analyze_latent_space_disentanglement(vae, config):
    """Analyze the disentanglement properties of the latent space"""
    print("Analyzing latent space disentanglement...")

    # Load data for analysis
    train_data = pd.read_csv(config['train_data_path'])
    feature_columns = [f'pixel_{i}' for i in range(config['input_dim'])]
    X_train = train_data[feature_columns].values
    y_train = train_data['label'].values if 'label' in train_data.columns else None

    # Create a figure to show traversal along each latent dimension
    vae.eval()

    # Number of samples to visualize per dimension
    n_samples = 10
    # Range of values to traverse
    z_range = np.linspace(-3, 3, n_samples)

    # Create a figure
    plt.figure(figsize=(15, config['latent_dim'] * 1.5))

    # For each latent dimension
    for dim in range(config['latent_dim']):
        # Get a random sample as a starting point
        random_idx = np.random.randint(0, len(X_train))
        x = torch.FloatTensor(X_train[random_idx:random_idx + 1])

        # Get its latent representation
        with torch.no_grad():
            mu, _, _, _ = vae(x)
            z_base = mu.clone()

        # Traverse along the current dimension
        traversals = []
        for val in z_range:
            # Modify only the current dimension
            z_new = z_base.clone()
            z_new[0, dim] = val

            # Decode
            with torch.no_grad():
                x_reconstructed = vae.decode(z_new)

            traversals.append(x_reconstructed.squeeze().numpy())

        # Plot the traversal
        for i, val in enumerate(z_range):
            plt.subplot(config['latent_dim'], n_samples, dim * n_samples + i + 1)
            plt.imshow(traversals[i].reshape(8, 9), cmap='gray')
            if i == 0:
                plt.ylabel(f"z{dim}")
            plt.xticks([])
            plt.yticks([])

    plt.tight_layout()
    plt.savefig(f"{config['save_dir']}/latent_disentanglement.png", dpi=300)
    plt.close()
    print(f"Disentanglement analysis saved to {config['save_dir']}/latent_disentanglement.png")


def generate_latent_space_animation(vae, config):
    """Generate animation of traversal through latent space"""
    print("Generating latent space animation...")

    # We'll create an animation rotating through a 2D plane in the latent space
    if config['latent_dim'] >= 2:
        # Number of frames
        n_frames = 50

        # Create a figure for animation
        fig, ax = plt.subplots(figsize=(6, 6))

        # Function to update the frame
        def update(frame):
            # Clear previous frame
            ax.clear()

            # Angle for this frame
            angle = frame / n_frames * 2 * np.pi

            # Create a latent vector with zeros except for the first two dimensions
            z = torch.zeros(1, config['latent_dim'])
            z[0, 0] = 3 * np.cos(angle)  # First dimension
            z[0, 1] = 3 * np.sin(angle)  # Second dimension

            # Decode the latent vector
            with torch.no_grad():
                x_reconstructed = vae.decode(z)

            # Plot the reconstructed image
            ax.imshow(x_reconstructed.squeeze().numpy().reshape(8, 9), cmap='gray')
            ax.set_title(f"Latent Space Rotation (z0={z[0, 0]:.2f}, z1={z[0, 1]:.2f})")
            ax.axis('off')

        # Create the animation
        ani = FuncAnimation(fig, update, frames=n_frames, interval=100)

        # Save the animation
        ani.save(f"{config['save_dir']}/latent_rotation.gif", writer='pillow', dpi=100)
        plt.close()

        print(f"Latent space animation saved to {config['save_dir']}/latent_rotation.gif")


if __name__ == "__main__":
    main()