import torch
import torch.nn as nn
import torch.optim as optim
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
from sklearn.manifold import TSNE
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


def create_detailed_visualizations(autoencoder, kmeans, encodings, X_train, y_train, config):

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
        'batch_size': 90000,
        'num_epochs': 1000,
        'learning_rate': 0.01,
        'input_dim': 72,
        'encoding1_dim': 20,
        'encoding2_dim': 10,
        'latent_dim': 10,
        'n_clusters': 10,
        # 'train_data_path': 'final_datasets/train_dataset.csv',
        'train_data_path': 'generated_data/gen_train_data.csv',
        'save_dir': './models/test3', #clustering
        'visualize': True
    }

    # Train and save model
    autoencoder, kmeans, cluster_labels = train_and_save_clustering_model(config)
    
    print("\nTraining and saving completed!")
    print(f"Models saved in {config['save_dir']}")
    if config['visualize']:
        print(f"Visualization saved as {config['save_dir']}/clustering_visualization.png")

if __name__ == "__main__":
    main()