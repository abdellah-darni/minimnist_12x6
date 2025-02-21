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
    def __init__(self, input_dim=72, encoding_dim=20, latent_dim=10):
        super(AutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(encoding_dim, latent_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, encoding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(encoding_dim, input_dim),
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
        encoding_dim=config['encoding_dim'],
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
        
            print(f'Epoch [{epoch+1}/{config["num_epochs"]}], Loss: {running_loss:.4f}')

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
        from sklearn.manifold import TSNE
        print("Creating visualization...")
        
        # Use t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=config['random_seed'])
        encodings_2d = tsne.fit_transform(encodings)
        
        # Plot results
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(encodings_2d[:, 0], encodings_2d[:, 1], 
                            c=cluster_labels, cmap='tab10')
        plt.colorbar(scatter)
        plt.title('Clustering Results (t-SNE visualization)')
        plt.savefig(f"{config['save_dir']}/clustering_visualization.png")
        plt.close()

    return autoencoder, kmeans, cluster_labels

def main():
    # Configuration
    config = {
        'random_seed': 42,
        'batch_size': 90000,
        'num_epochs': 500,
        'learning_rate': 0.00005,
        'input_dim': 72,
        'encoding_dim': 20,
        'latent_dim': 10,
        'n_clusters': 10,
        'train_data_path': 'final_datasets/train_dataset.csv',
        'save_dir': './models/clustering',
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