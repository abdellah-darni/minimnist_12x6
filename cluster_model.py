import os
import torch
import torch.nn as nn
import torch.optim as optim
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

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=72, encoding1_dim=20,encoding2_dim=10, latent_dim=10):
        super(AutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding1_dim),
            nn.ReLU(),
            nn.Linear(encoding1_dim, encoding2_dim),
            nn.ReLU(),
            nn.Linear(encoding2_dim, latent_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, encoding2_dim),
            nn.ReLU(),
            nn.Linear(encoding2_dim, encoding1_dim),
            nn.ReLU(),
            nn.Linear(encoding1_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def save_models(autoencoder, kmeans, save_dir):
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Save autoencoder
    torch.save(autoencoder.state_dict(), f"{save_dir}/autoencoder.pth")
    
    # Save KMeans
    with open(f"{save_dir}/kmeans.pkl", 'wb') as f:
        pickle.dump(kmeans, f)
    
    print(f"Models saved to {save_dir}")


def create_detailed_visualizations___no(autoencoder, kmeans, encodings, X_train, y_train, config):

    # 1. Cluster Composition Analysis
    plt.figure(figsize=(15, 6))
    
    # Count digits in each cluster
    cluster_labels = kmeans.labels_
    cluster_digit_composition = {}
    for cluster in range(config['n_clusters']):
        cluster_data = y_train[cluster_labels == cluster]
        unique, counts = np.unique(cluster_data, return_counts=True)
        cluster_digit_composition[cluster] = dict(zip(unique, counts))
    
    # Prepare data for stacked bar plot
    cluster_names = list(range(config['n_clusters']))
    digit_colors = plt.cm.tab10.colors
    
    plt.subplot(121)
    bottom = np.zeros(len(cluster_names))
    
    for digit in range(10):
        digit_counts = [cluster_digit_composition[cluster].get(digit, 0) for cluster in cluster_names]
        plt.bar(cluster_names, digit_counts, bottom=bottom, label=f'Digit {digit}', color=digit_colors[digit])
        bottom += digit_counts
    
    plt.title('Digit Composition in Each Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Samples')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Reconstruction Quality
    plt.subplot(122)
    
    # Compute reconstruction error for each cluster
    autoencoder.eval()
    reconstruction_errors = []
    
    for cluster in range(config['n_clusters']):
        cluster_mask = (cluster_labels == cluster)
        cluster_data = X_train[cluster_mask]
        
        # Convert to tensor
        cluster_tensor = torch.FloatTensor(cluster_data)
        
        # Compute reconstruction
        with torch.no_grad():
            _, reconstructed = autoencoder(cluster_tensor)
        
        # Compute mean squared error
        mse = np.mean(np.square(cluster_data - reconstructed.numpy()), axis=1)
        reconstruction_errors.append(np.mean(mse))
    
    plt.bar(cluster_names, reconstruction_errors, color='skyblue')
    plt.title('Reconstruction Error by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Mean Reconstruction Error')
    
    plt.tight_layout()
    plt.savefig(f"{config['save_dir']}/cluster_analysis.png")
    plt.close()

    # 3. Interactive 3D Visualization with Cluster Information
    # Perform t-SNE
    tsne_3d = TSNE(n_components=3, random_state=config['random_seed'])
    X_tsne_3d = tsne_3d.fit_transform(encodings)
    
    # Prepare hover text with detailed information
    hover_text = []
    for i, (cluster, digit) in enumerate(zip(cluster_labels, y_train)):
        # Count of each digit in this cluster
        digit_counts = cluster_digit_composition[cluster]
        digit_percent = digit_counts.get(digit, 0) / sum(digit_counts) * 100
        
        hover_info = (
            f"Cluster: {cluster}<br>"
            f"Digit: {digit}<br>"
            f"Cluster Composition:<br>"
            + "\n".join([f"Digit {d}: {count} ({count/sum(digit_counts)*100:.1f}%)" 
                         for d, count in sorted(digit_counts.items())])
        )
        hover_text.append(hover_info)
    
    # Interactive 3D plot
    fig = go.Figure(data=[go.Scatter3d(
        x=X_tsne_3d[:, 0],
        y=X_tsne_3d[:, 1],
        z=X_tsne_3d[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=cluster_labels,
            colorscale='Viridis',
            opacity=0.8
        ),
        text=hover_text,
        hoverinfo='text'
    )])

    fig.update_layout(
        title='3D Visualization of Clustering Results',
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3'
        )
    )

    # Save interactive plot
    fig.write_html(f"{config['save_dir']}/cluster_interactive_3d.html")

    print("Detailed Visualization Completed:")
    print(f"Cluster Analysis Plot: {config['save_dir']}/cluster_analysis.png")
    print(f"Interactive 3D Visualization: {config['save_dir']}/cluster_interactive_3d.html")


def create_detailed_visualizations(autoencoder, kmeans, encodings, X_train, y_train, config):

    # Try to import UMAP, but continue even if it's not available
    try:
        import umap
        has_umap = True
    except ImportError:
        has_umap = False
        print("UMAP not available, skipping UMAP visualizations")

    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    # Extract useful data
    cluster_labels = kmeans.labels_

    # Create color maps for digits and clusters
    digit_cmap = plt.get_cmap(name='tab10', lut=10)
    cluster_cmap = plt.get_cmap(name='viridis', lut=config['n_clusters'])

    # 1. Cluster composition analysis - Create a more detailed stacked bar chart
    plt.figure(figsize=(16, 8))
    cluster_digit_composition = {}
    for cluster in range(config['n_clusters']):
        cluster_data = y_train[cluster_labels == cluster]
        unique, counts = np.unique(cluster_data, return_counts=True)
        cluster_digit_composition[cluster] = dict(zip(unique, counts))

    # Prepare data for stacked bar plot
    cluster_names = list(range(config['n_clusters']))

    # Create a raw count version
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

    # Normalized version showing percentages
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
    gs = gridspec.GridSpec(2, 3 if has_umap else 2)

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

    ax3 = plt.subplot(gs[0, 2] if has_umap else gs[1, 0])
    scatter3 = ax3.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=y_train, cmap='tab10',
                           alpha=0.6, s=10)
    ax3.set_title('t-SNE - Digit Labels')
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')
    plt.colorbar(scatter3, ax=ax3, label='Digit')

    ax4 = plt.subplot(gs[1, 0] if has_umap else gs[1, 1])
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
    autoencoder.eval()

    n_rows = 10  # One row per digit
    n_cols = 10  # 5 Original and 5 reconstructions

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
        _, reconstructed = autoencoder(X_tensor)

    reconstructed = reconstructed.numpy()

    # Plot original and reconstructed digits
    for i, digit in enumerate(range(10)):
        digit_indices = [j for j, y in enumerate(y_samples) if y == digit]
        if not digit_indices:
            continue

        for j, idx in enumerate(digit_indices[:5]):
            # Original
            plt.subplot(n_rows, n_cols, i * n_cols + j + 1)
            # For 12x6 digits
            pixels = X_samples[idx].reshape(12, 6)
            plt.imshow(pixels, cmap='gray_r')
            plt.title(f"Original {digit}")
            plt.axis('off')

            # Reconstructed
            plt.subplot(n_rows, n_cols, i * n_cols + j + 1 + 5)
            pixels_recon = reconstructed[idx].reshape(12, 6)
            plt.imshow(pixels_recon, cmap='gray_r')
            plt.title(f"Reconstructed {digit}")
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/reconstruction_examples.png", dpi=300)
    plt.close()

    # 4. Latent space interpolation
    plt.figure(figsize=(15, 10))

    # Choose digit pairs to interpolate between
    digit_pairs = [(0, 1), (3, 8), (4, 9), (6, 8)]
    n_steps = 10

    autoencoder.eval()

    for pair_idx, (digit1, digit2) in enumerate(digit_pairs):
        # Get sample indices for the digit pair
        digit1_indices = np.where(y_train == digit1)[0]
        digit2_indices = np.where(y_train == digit2)[0]

        if len(digit1_indices) == 0 or len(digit2_indices) == 0:
            continue

        # Take first sample of each digit
        sample1_idx = np.random.choice(digit1_indices, 1)[0]
        sample2_idx = np.random.choice(digit2_indices, 1)[0]

        sample1 = torch.FloatTensor(X_train[sample1_idx]).unsqueeze(0)
        sample2 = torch.FloatTensor(X_train[sample2_idx]).unsqueeze(0)

        # Encode both samples to get latent vectors
        with torch.no_grad():
            z1, _ = autoencoder(sample1)
            z2, _ = autoencoder(sample2)

        # Create interpolations in latent space
        alphas = np.linspace(0, 1, n_steps)
        interpolations = []

        for alpha in alphas:
            # Interpolate between the latent vectors
            z_interp = alpha * z2 + (1 - alpha) * z1

            # Decode the interpolated latent vector
            with torch.no_grad():
                decoded = autoencoder.decoder(z_interp)

            interpolations.append(decoded.squeeze().numpy())

        # Plot the interpolations
        for i, interp in enumerate(interpolations):
            plt.subplot(len(digit_pairs), n_steps, pair_idx * n_steps + i + 1)
            plt.imshow(interp.reshape(12, 6), cmap='gray_r')
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
            generated = autoencoder.decoder(z_tensor)

        generated = generated.numpy()

        # Plot the generated digits
        for i in range(n_samples):
            plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(generated[i].reshape(12, 6), cmap='gray_r')
            plt.axis('off')

        plt.suptitle("Generated Digits from 2D Latent Space Grid", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/latent_space_generation.png", dpi=300)
        plt.close()

    # # 6. Interactive 3D visualization
    # # Perform t-SNE to 3D
    # tsne_3d = TSNE(n_components=3, random_state=config['random_seed'])
    # latent_tsne_3d = tsne_3d.fit_transform(latent_vecs)
    #
    # # Create dual visualizations - one by digit, one by cluster
    # fig = make_subplots(
    #     rows=1, cols=2,
    #     specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
    #     subplot_titles=('Colored by Digit', 'Colored by Cluster')
    # )
    #
    # # First plot - colored by digit
    # fig.add_trace(
    #     go.Scatter3d(
    #         x=latent_tsne_3d[:, 0],
    #         y=latent_tsne_3d[:, 1],
    #         z=latent_tsne_3d[:, 2],
    #         mode='markers',
    #         marker=dict(
    #             size=4,
    #             color=y_train,
    #             colorscale='Turbo',
    #             opacity=0.8,
    #             colorbar=dict(
    #                 title="Digit",
    #                 x=0.45
    #             )
    #         ),
    #         text=[f"Digit: {y}<br>Cluster: {c}" for y, c in zip(y_train, cluster_labels)],
    #         hoverinfo='text'
    #     ),
    #     row=1, col=1
    # )
    #
    # # Second plot - colored by cluster
    # fig.add_trace(
    #     go.Scatter3d(
    #         x=latent_tsne_3d[:, 0],
    #         y=latent_tsne_3d[:, 1],
    #         z=latent_tsne_3d[:, 2],
    #         mode='markers',
    #         marker=dict(
    #             size=4,
    #             color=cluster_labels,
    #             colorscale='Viridis',
    #             opacity=0.8,
    #             colorbar=dict(
    #                 title="Cluster",
    #                 x=1.0
    #             )
    #         ),
    #         text=[f"Digit: {y}<br>Cluster: {c}" for y, c in zip(y_train, cluster_labels)],
    #         hoverinfo='text'
    #     ),
    #     row=1, col=2
    # )
    #
    # fig.update_layout(
    #     title='3D Visualization of Autoencoder Latent Space',
    #     height=800,
    #     scene=dict(
    #         xaxis_title='t-SNE 1',
    #         yaxis_title='t-SNE 2',
    #         zaxis_title='t-SNE 3'
    #     ),
    #     scene2=dict(
    #         xaxis_title='t-SNE 1',
    #         yaxis_title='t-SNE 2',
    #         zaxis_title='t-SNE 3'
    #     )
    # )
    #
    # # Save interactive plot
    # fig.write_html(f"{save_dir}/3d_latent_visualization.html")
    #
    # 7. Create a latent space traversal visualization if latent dim >= 2
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
            generated = autoencoder.decoder(z)

        # Plot the traversals
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                plt.subplot(grid_size, grid_size, idx + 1)
                plt.imshow(generated[idx].numpy().reshape(12, 6), cmap='gray_r')
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
        centroid_decoded = autoencoder.decoder(torch.FloatTensor(centroids))

    # Plot the centroid reconstructions
    for i in range(config['n_clusters']):
        plt.subplot(2, (config['n_clusters'] + 1) // 2, i + 1)
        plt.imshow(centroid_decoded[i].numpy().reshape(12, 6), cmap='gray_r')
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
    if config['latent_dim'] >= 2:
        print(f"- Latent Space Generation: {save_dir}/latent_space_generation.png")
        print(f"- Latent Space Traversal: {save_dir}/latent_traversal.png")
    print(f"- Interactive 3D Visualization: {save_dir}/3d_latent_visualization.html")
    print(f"- Cluster Centroids: {save_dir}/cluster_centroids.png")


def train_and_save_clustering_model(config):
    
    # Set seeds and device
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    try:
        train_data = pd.read_csv(config['train_data_path'])
        feature_columns = [f'pixel_{i}' for i in range(72)]
        X_train = train_data[feature_columns].values
        y_train = train_data['label'].values if 'label' in train_data.columns else None
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Create dataset and dataloader
    dataset = DigitDataset(X_train, y_train)
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # Initialize model and optimizer
    print("Initializing model...")
    autoencoder = AutoEncoder(
        input_dim=config['input_dim'],
        encoding1_dim=config['encoding1_dim'],
        encoding2_dim=config['encoding2_dim'],
        latent_dim=config['latent_dim']
    ).to(device)
    
    optimizer = optim.Adam(autoencoder.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()

    # Train autoencoder
    print("Training autoencoder...")
    autoencoder.train()
    for epoch in range(config['num_epochs']):
        running_loss = 0.0
        for batch_features, _ in data_loader:
            batch_features = batch_features.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            _, reconstructed = autoencoder(batch_features)
            loss = criterion(reconstructed, batch_features)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
            # Print progress
        # if (epoch + 1) % 50 == 0:
        #     avg_loss = running_loss / len(data_loader)
        #     print(f'Epoch [{epoch+1}/{config["num_epochs"]}], Loss: {avg_loss:.4f}')

        if (epoch + 1) % 5 == 0:
            avg_loss = running_loss / len(data_loader)
            print(f'Epoch [{epoch+1}/{config["num_epochs"]}], Loss: {avg_loss:.4f}')

    # Get encodings for clustering
    print("Performing clustering...")
    autoencoder.eval()
    encodings = []
    with torch.no_grad():
        for batch_features, _ in data_loader:
            batch_features = batch_features.to(device)
            encoded, _ = autoencoder(batch_features)
            encodings.append(encoded.cpu().numpy())
    
    encodings = np.concatenate(encodings)

    # Perform clustering
    kmeans = KMeans(n_clusters=config['n_clusters'], random_state=config['random_seed'])
    cluster_labels = kmeans.fit_predict(encodings)

    # Save models
    print("Saving models...")
    save_models(autoencoder, kmeans, config['save_dir'])

    # Visualize results
    if config['visualize']:
        
        create_detailed_visualizations(
            autoencoder, 
            kmeans, 
            encodings, 
            X_train, 
            y_train, 
            config
        )
  
    return autoencoder, kmeans, cluster_labels


def main():
    # Configuration
    config = {
        'random_seed': 42,
        'batch_size': 512,
        'num_epochs': 200,
        'learning_rate': 0.001,
        'input_dim': 72,
        'encoding1_dim': 20,
        'encoding2_dim': 10,
        'latent_dim': 5,
        'n_clusters': 10,
        # 'train_data_path': 'final_datasets/train_dataset.csv',
        'train_data_path': 'new_data/gen_train_data.csv',
        'save_dir': './models/new_cl',  # clustering
        'visualize': True
    }

    # Create save directory if it doesn't exist
    os.makedirs(config['save_dir'], exist_ok=True)

    # Train and save model
    autoencoder, kmeans, cluster_labels = train_and_save_clustering_model(config)

    print("\nTraining and saving completed!")
    print(f"Models saved in {config['save_dir']}")
    if config['visualize']:
        print(f"Visualizations saved in {config['save_dir']}")


if __name__ == "__main__":
    main()