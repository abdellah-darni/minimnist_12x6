import pandas as pd
import numpy as np


def binarize_image(image, threshold=65):  # i did find that 65 is the best threshold for our case 
    
    return (image > threshold).astype(np.uint8)


def crop_image(image, margin=1):
    

    rows = np.any(image, axis=1)
    cols = np.any(image, axis=0)
    
    if not rows.any() or not cols.any():
        # No nonzero pixels found; return the original image.
        return image

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add a margin around the bounding box
    rmin = max(rmin - margin, 0)
    rmax = min(rmax + margin, image.shape[0] - 1)
    cmin = max(cmin - margin, 0)
    cmax = min(cmax + margin, image.shape[1] - 1)

    
    cropped = image[rmin:rmax+1, cmin:cmax+1]
    return cropped


def resize_image(image, new_shape=(12, 6)):
    
    new_height, new_width = new_shape
    old_height, old_width = image.shape

    # Generate indices mapping from the new image to the original image.
    row_indices = (np.linspace(0, old_height, new_height, endpoint=False)).astype(int)
    col_indices = (np.linspace(0, old_width, new_width, endpoint=False)).astype(int)

    # Use np.ix_ to form a grid and extract the corresponding pixels.
    resized = image[np.ix_(row_indices, col_indices)]
    return resized


def process_image(flat_pixels):
    
    # Reshape the flat vector into a 28x28 image.
    image = flat_pixels.reshape(28, 28)
    
    # Convert to black and white (binary) image.
    binary = binarize_image(image)
    
    # Crop the extra border.
    cropped = crop_image(binary)
    
    # Resize the cropped image to 12x6.
    resized = resize_image(cropped, new_shape=(12, 6))
    return resized


def process_and_save_dataset(intput_file, output_file):

    df = pd.read_csv(intput_file)
    
    pixel_columns = df.columns[1:]
    
    processed_rows = []
    
    for idx, row in df.iterrows():
        # Extract the label
        label = row[df.columns[0]]
        
        # Convert the pixel values to a NumPy array.
        flat_pixels = row[pixel_columns].values.astype(np.uint8)
        
        # Process the image: binarize, crop, and resize.
        processed_img = process_image(flat_pixels)
        
        # Flatten the final 12x6 image to store in a single row.
        flattened_pixels = processed_img.flatten().tolist()
        
        # Append the label as the last element.
        flattened_pixels.append(label)
        
        processed_rows.append(flattened_pixels)
    
    col_names = [f'pixel_{i}' for i in range(0, 72)] + ['label']
    
    processed_df = pd.DataFrame(processed_rows, columns=col_names)
    
    processed_df.to_csv(output_file, index=False)
    print(f"Processing complete. Processed data saved to {output_file}.")


def main():
    
    # training data
    process_and_save_dataset(
        'mnist_data/original/mnist_train.csv',
        'mnist_data/processed/processed_mnist_train.csv'
    )

    # testing data
    process_and_save_dataset(
        'mnist_data/original/mnist_test.csv',
        'mnist_data/processed/processed_mnist_test.csv'
    )

if __name__ == '__main__':
    main()
