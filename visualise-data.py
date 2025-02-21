import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def visualize_digit_csv(csv_file, n_examples=10):
    
    # Load CSV data
    df = pd.read_csv(csv_file)

    # Define the pixel column names ( 72 pixels for a 12x6 grid)
    pixel_columns = [f"pixel_{i}" for i in range(72)]

    # Sample some examples (if the requested number is more than available, use all)
    n_examples = min(n_examples, len(df))
    sample_df = df.sample(n_examples)
    # sample_df = df.head(n_examples) #for templates

    # Setup the plot grid
    n_cols = 10
    n_rows = (n_examples + n_cols - 1) // n_cols  # ceiling division
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axs = axs.flatten()

    # Plot each sample
    for i, (_, row) in enumerate(sample_df.iterrows()):
        # Extract pixel values and reshape to 12x6 grid
        pixel_values = row[pixel_columns].values.astype(np.int64)
        image = pixel_values.reshape((12, 6))

        # Display the image
        axs[i].imshow(image, cmap="gray_r", interpolation="nearest")
        axs[i].axis("off")

        # Label
        title = f"\nNB:{row['label']}"
        axs[i].set_title(title, fontsize=6)

    # Turn off any extra axes if needed
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # generated dataset 

    # for training 
    # csv_file = "./generated_data/train/gen_train_data.csv"

    # for testing 
    # csv_file = "./generated_data/test/gen_test_data.csv"

    # processed mnist dataset
    # training
    # csv_file = 'mnist_data/processed/processed_mnist_train.csv'

    # testing
    # csv_file = 'mnist_data/processed/processed_mnist_test.csv'

    # the actual datasets
    # training 
    csv_file = 'final_datasets/train_dataset.csv'

    # testing 
    # csv_file = 'final_datasets/test_dataset.csv'


    visualize_digit_csv(csv_file, n_examples=100)
