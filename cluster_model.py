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


def create_detailed_visualizations(autoencoder, kmeans, encodings, X_train, y_train, config, input_samples=None):
    """
    Create detailed visualizations of the autoencoder and clustering results with enhanced capabilities:
    - Latent space dimension analysis
    - Input-to-output visualization
    - Direct latent space manipulation

    Parameters:
    -----------
    autoencoder : AutoEncoder model
        The trained autoencoder model
    kmeans : KMeans model
        The trained clustering model
    encodings : numpy.ndarray
        The latent space encodings for the training data
    X_train : numpy.ndarray
        The training data features
    y_train : numpy.ndarray
        The training data labels
    config : dict
        Configuration parameters
    input_samples : numpy.ndarray, optional
        Custom input samples to visualize (if None, samples from training data are used)
    """

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
    digit_cmap = plt.cm.get_cmap(name='tab10', lut=10)
    cluster_cmap = plt.cm.get_cmap(name='viridis', lut=config['n_clusters'])

    # ---------------------------------------------------
    # 1. Cluster composition analysis
    # ---------------------------------------------------
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

    # ---------------------------------------------------
    # 2. Latent space visualization with multiple techniques
    # ---------------------------------------------------
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

    # ---------------------------------------------------
    # 3. NEW: Analyzing what each latent dimension represents
    # ---------------------------------------------------
    latent_dim = config['latent_dim']
    plt.figure(figsize=(20, 15))

    # 3.1 Correlation of each latent dimension with digit classes
    # Create one-hot encoding for digits
    digit_onehot = np.zeros((len(y_train), 10))
    for i in range(len(y_train)):
        digit_onehot[i, y_train[i]] = 1

    # Calculate correlation between each latent dimension and each digit
    correlations = np.zeros((latent_dim, 10))
    p_values = np.zeros((latent_dim, 10))

    for dim in range(latent_dim):
        for digit in range(10):
            corr, p_val = pearsonr(latent_vecs[:, dim], digit_onehot[:, digit])
            correlations[dim, digit] = corr
            p_values[dim, digit] = p_val

    # Plot correlation heatmap
    plt.subplot(121)
    sns.heatmap(correlations, annot=True, cmap='coolwarm',
                xticklabels=range(10), yticklabels=[f'Dim {i}' for i in range(latent_dim)])
    plt.title('Correlation: Latent Dimensions vs Digit Classes')
    plt.xlabel('Digit')
    plt.ylabel('Latent Dimension')

    # 3.2 Plot latent dimension distributions by digit
    dim_digit_data = []
    for dim in range(min(5, latent_dim)):  # Show top 5 dimensions only
        plt.subplot(2, 3, dim + 2)
        for digit in range(10):
            dim_values = latent_vecs[y_train == digit, dim]
            sns.kdeplot(dim_values, label=f'Digit {digit}')

            # Also collect data for the interactive plot
            for val in dim_values:
                dim_digit_data.append({
                    'dimension': f'Dimension {dim}',
                    'value': val,
                    'digit': f'Digit {digit}'
                })

        plt.title(f'Distribution of Latent Dimension {dim} by Digit')
        plt.xlabel(f'Dimension {dim} Value')
        plt.ylabel('Density')
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/latent_dimension_analysis.png", dpi=300)
    plt.close()

    # Create interactive boxplot for all dimensions vs digits
    dim_digit_df = pd.DataFrame(dim_digit_data)
    fig = px.box(dim_digit_df, x='digit', y='value', color='digit',
                 facet_col='dimension', facet_col_wrap=2,
                 title='Distribution of Latent Space Dimensions by Digit')
    fig.update_layout(height=1000, width=1000)
    fig.write_html(f"{save_dir}/latent_boxplots_by_digit.html")

    # ---------------------------------------------------
    # 4. NEW: Input-to-output visualization with samples
    # ---------------------------------------------------

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
    autoencoder.eval()
    X_tensor = torch.FloatTensor(X_samples)

    with torch.no_grad():
        encoded, reconstructed = autoencoder(X_tensor)

    encoded = encoded.numpy()
    reconstructed = reconstructed.numpy()

    # Create a figure showing original, encoding, and reconstruction
    fig = plt.figure(figsize=(15, len(X_samples)))
    gs = gridspec.GridSpec(len(X_samples), 3)

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

        # Latent encoding (bar chart)
        ax2 = plt.subplot(gs[i, 1])
        ax2.bar(range(latent_dim), encoded[i], color='steelblue')
        if i == 0:
            ax2.set_title('Latent Space Encoding')
        ax2.set_xticks(range(latent_dim))
        ax2.set_xticklabels([f'D{j}' for j in range(latent_dim)], rotation=45)

        # Reconstruction
        ax3 = plt.subplot(gs[i, 2])
        ax3.imshow(reconstructed[i].reshape(12, 6), cmap='gray_r')
        if i == 0:
            ax3.set_title('Reconstruction')
        ax3.set_xticks([])
        ax3.set_yticks([])

    plt.tight_layout()
    plt.savefig(f"{save_dir}/input_output_visualization.png", dpi=300)
    plt.close()

    # ---------------------------------------------------
    # 5. NEW: Interactive latent space manipulation
    # ---------------------------------------------------
    # Create a figure with sliders to manipulate latent dimensions

    # Generate an initial latent vector (average of all encoded vectors)
    initial_latent = np.mean(encodings, axis=0)

    # Create interactive visualization with Plotly
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Latent Space Sliders', 'Generated Output'),
                        specs=[[{'type': 'domain'}, {'type': 'heatmap'}]])

    # Create a starting image
    with torch.no_grad():
        initial_output = autoencoder.decoder(torch.FloatTensor(initial_latent).unsqueeze(0))
        initial_output = initial_output.squeeze().numpy().reshape(12, 6)

    # Add the heatmap for the digit image
    heatmap = go.Heatmap(z=initial_output, colorscale='gray_r', showscale=False)
    fig.add_trace(heatmap, row=1, col=2)

    # Create a DataFrame for the sliders
    slider_data = []
    for dim in range(latent_dim):
        # Find min/max values in the dataset for this dimension
        dim_min = np.min(encodings[:, dim])
        dim_max = np.max(encodings[:, dim])

        # Add a margin
        range_width = dim_max - dim_min
        dim_min -= range_width * 0.2
        dim_max += range_width * 0.2

        # Get correlations with digits for this dimension
        dim_correlations = correlations[dim]
        most_correlated_digit = np.argmax(np.abs(dim_correlations))
        correlation_strength = dim_correlations[most_correlated_digit]

        # Add to slider data
        slider_data.append({
            'dimension': f'Dimension {dim}',
            'value': initial_latent[dim],
            'min': dim_min,
            'max': dim_max,
            'most_correlated_digit': int(most_correlated_digit),
            'correlation': correlation_strength
        })

    # Create DataFrame
    slider_df = pd.DataFrame(slider_data)

    # Create a table for latent dimensions
    table = go.Table(
        header=dict(
            values=['Dimension', 'Value', 'Most Correlated Digit', 'Correlation'],
            font=dict(size=12),
            align='left'
        ),
        cells=dict(
            values=[
                slider_df['dimension'],
                slider_df['value'].round(2),
                slider_df['most_correlated_digit'],
                slider_df['correlation'].round(2)
            ],
            font=dict(size=11),
            align='left'
        )
    )

    fig.add_trace(table, row=1, col=1)

    # Add sliders to control latent dimensions
    sliders = []
    for i, row in slider_df.iterrows():
        sliders.append(
            dict(
                active=50,  # middle position
                currentvalue={"prefix": f"Dimension {i}: "},
                pad={"t": 30 + i * 20},
                steps=[
                    dict(
                        method='restyle',
                        args=[
                            {},
                            {}
                        ],
                        label=str(val)
                    )
                    for val in np.linspace(row['min'], row['max'], 100)
                ]
            )
        )

    fig.update_layout(
        title_text="Interactive Latent Space Manipulation",
        height=800,
        width=1200,
        sliders=sliders
    )

    # Add instructions and save
    with open(f"{save_dir}/latent_manipulation.html", 'w') as f:
        f.write("""
        <html>
        <head>
            <title>Latent Space Manipulation</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>Interactive Latent Space Manipulation</h1>
            <p>This tool lets you manually manipulate each dimension of the latent space to see how it affects the output.</p>
            <p>Instructions:</p>
            <ol>
                <li>Change the sliders to manipulate different latent dimensions</li>
                <li>The image on the right will update to show what digit the model generates</li>
                <li>The table shows which digit each dimension is most correlated with</li>
            </ol>

            <div id="latent-plot"></div>

            <script>
                // Load pre-computed encodings and model
                const initialLatent = """ + str(initial_latent.tolist()) + """;
                let currentLatent = [...initialLatent];

                // Create the initial plot
                const plotDiv = document.getElementById('latent-plot');

                // Create the plot layout
                const layout = {
                    title: 'Latent Space Manipulation',
                    width: 1000,
                    height: 600,
                    grid: {rows: 1, columns: 2},
                    sliders: []
                };

                // Create sliders for each dimension
                """ + "\n".join([
            f"""
                    layout.sliders.push({{
                        x: 0.1,
                        y: {0.9 - i * 0.05},
                        len: 0.8,
                        currentvalue: {{
                            prefix: 'Dimension {i}: ',
                            xanchor: 'right'
                        }},
                        steps: Array.from({{ length: 100 }}, (_, j) => {{
                            const val = {row['min']} + ({row['max']} - {row['min']}) * j / 99;
                            return {{
                                label: val.toFixed(2),
                                method: 'animate',
                                args: [[{{value: val}}]]
                            }};
                        }})
                    }});
                    """
            for i, row in slider_df.iterrows()
        ]) + """

                // TO BE COMPLETED: Add JavaScript to handle slider interactions and update the image
                // This would require serving this as a web application with a backend
                // For static HTML export, we'll include instructions

                // Create a placeholder plot
                Plotly.newPlot(plotDiv, [{
                    type: 'heatmap',
                    z: """ + str(initial_output.tolist()) + """,
                    colorscale: 'Greys'
                }], layout);
            </script>

            <p><b>Note:</b> Full interactive manipulation requires a web server with Python backend. 
            For a simpler version, see the static visualizations of latent space traversal.</p>
        </body>
        </html>
        """)

    # 6. Improved latent space traversal visualization
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
                    generated = autoencoder.decoder(torch.FloatTensor(latent).unsqueeze(0))
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

    # ---------------------------------------------------
    # 7. Function to process custom inputs and generate outputs
    # ---------------------------------------------------
    def process_custom_input(input_data, output_file=None):
        """
        Process a custom input through the autoencoder and visualize results

        Parameters:
        -----------
        input_data : numpy.ndarray
            Input data of shape (72,) or (n_samples, 72)
        output_file : str, optional
            If provided, save the visualization to this file

        Returns:
        --------
        dict : Results containing original, encoding, and reconstruction
        """
        # Reshape input if needed
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)

        # Convert to tensor
        input_tensor = torch.FloatTensor(input_data)

        # Process through autoencoder
        with torch.no_grad():
            encoded, reconstructed = autoencoder(input_tensor)

        # Convert to numpy
        encoded_np = encoded.numpy()
        reconstructed_np = reconstructed.numpy()

        # Create visualization
        n_samples = input_data.shape[0]
        fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))

        # Handle case with single sample
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            # Original
            axes[i, 0].imshow(input_data[i].reshape(12, 6), cmap='gray_r')
            axes[i, 0].set_title('Original Input')
            axes[i, 0].axis('off')

            # Latent encoding
            axes[i, 1].bar(range(config['latent_dim']), encoded_np[i])
            axes[i, 1].set_title('Latent Space Encoding')
            axes[i, 1].set_xlabel('Dimension')
            axes[i, 1].set_xticks(range(config['latent_dim']))

            # Reconstruction
            axes[i, 2].imshow(reconstructed_np[i].reshape(12, 6), cmap='gray_r')
            axes[i, 2].set_title('Reconstruction')
            axes[i, 2].axis('off')

        plt.tight_layout()

        # Save if output file is provided
        if output_file:
            plt.savefig(output_file, dpi=300)
            plt.close()
        else:
            plt.show()

        return {
            'original': input_data,
            'encoded': encoded_np,
            'reconstructed': reconstructed_np
        }

    # ---------------------------------------------------
    # 8. Function to generate output from custom latent vector
    # ---------------------------------------------------
    def generate_from_latent(latent_vector, output_file=None):
        """
        Generate output directly from a latent vector

        Parameters:
        -----------
        latent_vector : numpy.ndarray
            Latent vector of shape (latent_dim,) or (n_samples, latent_dim)
        output_file : str, optional
            If provided, save the visualization to this file

        Returns:
        --------
        numpy.ndarray : Generated output(s)
        """
        # Reshape if needed
        if latent_vector.ndim == 1:
            latent_vector = latent_vector.reshape(1, -1)

        # Convert to tensor
        latent_tensor = torch.FloatTensor(latent_vector)

        # Generate output
        with torch.no_grad():
            generated = autoencoder.decoder(latent_tensor)

        # Convert to numpy
        generated_np = generated.numpy()

        # Create visualization
        n_samples = latent_vector.shape[0]
        fig, axes = plt.subplots(1, n_samples, figsize=(5 * n_samples, 5))

        # Handle case with single sample
        if n_samples == 1:
            axes = np.array([axes])

        for i in range(n_samples):
            axes[i].imshow(generated_np[i].reshape(12, 6), cmap='gray_r')
            axes[i].set_title(f'Generated from latent vector {i + 1}')
            axes[i].axis('off')

        plt.tight_layout()

        # Save if output file is provided
        if output_file:
            plt.savefig(output_file, dpi=300)
            plt.close()
        else:
            plt.show()

        return generated_np

    # Create sample use case for custom input
    sample_input = X_train[y_train == 5][0:3]  # Get 3 examples of digit 5
    process_custom_input(sample_input, f"{save_dir}/sample_custom_input.png")

    # Create sample use case for latent vector manipulation
    # Generate a "mixed" digit by interpolating between latent vectors
    if y_train is not None:
        # Get average latent vector for digit 3 and digit 8
        digit_3_latent = np.mean(encodings[y_train == 3], axis=0)
        digit_8_latent = np.mean(encodings[y_train == 8], axis=0)

        # Create an interpolation
        mixed_latent = 0.5 * digit_3_latent + 0.5 * digit_8_latent

        # Generate from these latent vectors
        sample_latents = np.vstack([digit_3_latent, mixed_latent, digit_8_latent])
        generate_from_latent(sample_latents, f"{save_dir}/sample_latent_manipulation.png")


def train_semi_supervised_autoencoder(config):
    # Set seeds and device
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    train_data = pd.read_csv(config['train_data_path'])
    feature_columns = [f'pixel_{i}' for i in range(72)]
    X_train = train_data[feature_columns].values
    y_train = train_data['label'].values if 'label' in train_data.columns else None

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
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.to(device)

            # Forward pass
            optimizer.zero_grad()
            encoded, reconstructed = autoencoder(batch_features)

            # Reconstruction loss
            recon_loss = criterion(reconstructed, batch_features)

            # Clustering guidance loss (if we have labels)
            if batch_labels is not None and config['use_label_guidance']:
                # Calculate pairwise distances in latent space
                pairwise_dist = torch.cdist(encoded, encoded, p=2)

                # Create a mask for same-label pairs (1 if same label, 0 if different)
                label_matrix = batch_labels.unsqueeze(0) == batch_labels.unsqueeze(1)

                # Convert to float and to device
                label_matrix = label_matrix.float().to(device)

                # Calculate contrastive loss
                # Minimize distance for same label, maximize for different labels
                margin = config['margin']

                # For same label pairs: distance should be small
                same_label_loss = label_matrix * pairwise_dist

                # For different label pairs: distance should be at least margin
                diff_label_loss = (1 - label_matrix) * torch.clamp(margin - pairwise_dist, min=0)

                # Combine both components
                contrastive_loss = (same_label_loss + diff_label_loss).mean()

                # Total loss with weighting
                loss = recon_loss + config['contrastive_weight'] * contrastive_loss
            else:
                loss = recon_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print progress
        if (epoch + 1) % 5 == 0:
            avg_loss = running_loss / len(data_loader)
            print(f'Epoch [{epoch + 1}/{config["num_epochs"]}], Loss: {avg_loss:.4f}')

    # Get encodings
    print("Extracting encodings...")
    autoencoder.eval()
    all_encodings = []
    all_labels = []

    with torch.no_grad():
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.to(device)
            encoded, _ = autoencoder(batch_features)
            all_encodings.append(encoded.cpu().numpy())
            if batch_labels is not None:
                all_labels.append(batch_labels.numpy())

    encodings = np.concatenate(all_encodings)
    labels = np.concatenate(all_labels) if all_labels else None

    # Perform clustering
    print("Performing clustering...")

    if config['clustering_method'] == 'kmeans':
        # Standard KMeans
        kmeans = KMeans(n_clusters=config['n_clusters'], random_state=config['random_seed'])
        cluster_labels = kmeans.fit_predict(encodings)
        clustering_model = kmeans

    elif config['clustering_method'] == 'gmm':
        # Gaussian Mixture Model
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(
            n_components=config['n_clusters'],
            covariance_type='full',
            random_state=config['random_seed'],
            max_iter=100
        )
        cluster_labels = gmm.fit_predict(encodings)
        clustering_model = gmm

    elif config['clustering_method'] == 'dbscan':
        # DBSCAN
        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(eps=config['dbscan_eps'], min_samples=config['dbscan_min_samples'])
        cluster_labels = dbscan.fit_predict(encodings)
        clustering_model = dbscan

    elif config['clustering_method'] == 'semi_supervised':
        # Semi-supervised clustering with label propagation
        from sklearn.semi_supervised import LabelPropagation

        # Use a small portion of labels to guide clustering
        if labels is not None:
            label_mask = np.zeros(len(labels), dtype=bool)
            for digit in range(10):
                # Take some samples from each digit
                digit_indices = np.where(labels == digit)[0]
                if len(digit_indices) > 0:
                    # Use 10% of each digit's samples (or at least 1)
                    n_samples = max(1, int(len(digit_indices) * 0.1))
                    label_indices = np.random.choice(digit_indices, n_samples, replace=False)
                    label_mask[label_indices] = True

            # Create semi-supervised labels: -1 for unlabeled points
            semi_labels = np.full(len(labels), -1)
            semi_labels[label_mask] = labels[label_mask]

            # Label propagation
            model = LabelPropagation()
            cluster_labels = model.fit(encodings, semi_labels).predict(encodings)
            clustering_model = model
        else:
            # Fallback to KMeans if no labels
            kmeans = KMeans(n_clusters=config['n_clusters'], random_state=config['random_seed'])
            cluster_labels = kmeans.fit_predict(encodings)
            clustering_model = kmeans

    else:
        # Default to KMeans
        kmeans = KMeans(n_clusters=config['n_clusters'], random_state=config['random_seed'])
        cluster_labels = kmeans.fit_predict(encodings)
        clustering_model = kmeans

    # Save model and return results
    save_models(autoencoder, clustering_model, config['save_dir'])

    return autoencoder, clustering_model, encodings, cluster_labels, labels


# def main():
#     # Configuration
#     config = {
#         'random_seed': 42,
#         'batch_size': 512,
#         'num_epochs': 200,
#         'learning_rate': 0.001,
#         'input_dim': 72,
#         'encoding1_dim': 64,
#         'encoding2_dim': 32,
#         'latent_dim': 10,
#         'n_clusters': 10,
#         # 'train_data_path': 'final_datasets/train_dataset.csv',
#         'train_data_path': 'new_data/gen_train_data.csv',
#         'save_dir': './models/new_cl',  # clustering
#         'visualize': True
#     }
#
#     # Create save directory if it doesn't exist
#     os.makedirs(config['save_dir'], exist_ok=True)
#
#     # Train and save model
#     autoencoder, kmeans, cluster_labels = train_and_save_clustering_model(config)
#
#     print("\nTraining and saving completed!")
#     print(f"Models saved in {config['save_dir']}")
#     if config['visualize']:
#         print(f"Visualizations saved in {config['save_dir']}")


def main():
    # Updated configuration
    config = {
        'random_seed': 42,
        'batch_size': 128,  # Reduced batch size for better generalization
        'num_epochs': 300,  # Increased epochs
        'learning_rate': 0.001,
        'input_dim': 72,
        'encoding1_dim': 64,  # Wider network
        'encoding2_dim': 32,  # Wider network
        'latent_dim': 10,  # Match number of digits
        'n_clusters': 10,
        'train_data_path': 'new_data/gen_train_data.csv',
        'save_dir': './models/improved_clustering',
        'visualize': True,

        # New parameters for semi-supervised learning
        'use_label_guidance': True,
        'contrastive_weight': 0.5,
        'margin': 2.0,

        # Clustering method options
        'clustering_method': 'semi_supervised',  # Options: 'kmeans', 'gmm', 'dbscan', 'semi_supervised'
        'dbscan_eps': 0.5,
        'dbscan_min_samples': 5
    }

    # Create save directory if it doesn't exist
    os.makedirs(config['save_dir'], exist_ok=True)

    # Train and save model with semi-supervised approach
    autoencoder, clustering_model, encodings, cluster_labels, true_labels = train_semi_supervised_autoencoder(config)

    # Evaluate clustering performance if labels are available
    if true_labels is not None:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

        ari = adjusted_rand_score(true_labels, cluster_labels)
        nmi = normalized_mutual_info_score(true_labels, cluster_labels)

        try:
            silhouette = silhouette_score(encodings, cluster_labels)
        except:
            silhouette = "N/A"  # Some clustering methods might produce -1 labels for noise

        print("\nClustering Evaluation:")
        print(f"Adjusted Rand Index: {ari:.4f}")
        print(f"Normalized Mutual Information: {nmi:.4f}")
        print(f"Silhouette Score: {silhouette}")

        # Create a mapping from clusters to majority digit
        cluster_to_digit = {}
        for cluster in range(config['n_clusters']):
            cluster_indices = np.where(cluster_labels == cluster)[0]
            if len(cluster_indices) > 0:
                cluster_digits = true_labels[cluster_indices]
                digit_counts = np.bincount(cluster_digits)
                majority_digit = np.argmax(digit_counts)
                purity = digit_counts[majority_digit] / len(cluster_indices)
                cluster_to_digit[cluster] = (majority_digit, purity)
                print(f"Cluster {cluster} → Digit {majority_digit} (Purity: {purity:.2f})")

    # Visualize results
    if config['visualize']:
        create_detailed_visualizations(
            autoencoder,
            clustering_model,
            encodings,
            X_train,
            true_labels,
            config
        )

    print("\nTraining and saving completed!")
    print(f"Models saved in {config['save_dir']}")
    if config['visualize']:
        print(f"Visualizations saved in {config['save_dir']}")



if __name__ == "__main__":
    main()