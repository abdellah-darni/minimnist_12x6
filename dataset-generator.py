import numpy as np
import pandas as pd
import os


def get_templates_for_digit(digit):

    templates = []
    if digit == 0:

        t1 = np.array([ 
            [1,1,1,1,1,1],
            [1,1,0,0,1,1],
            [1,1,0,0,1,1],
            [1,1,0,0,1,1],
            [1,1,0,0,1,1],
            [1,1,0,0,1,1],
            [1,1,0,0,1,1],
            [1,1,0,0,1,1],
            [1,1,0,0,1,1],
            [1,1,0,0,1,1],
            [1,1,0,0,1,1],
            [1,1,1,1,1,1]
        ], dtype=np.float32)
        templates.append(t1)

        # t2 = np.array([
        #     [0,1,1,1,1,0],
        #     [1,1,0,0,1,1],
        #     [1,1,0,0,1,1],
        #     [1,1,0,0,1,1],
        #     [1,1,0,0,1,1],
        #     [1,1,0,0,1,1],
        #     [1,1,0,0,1,1],
        #     [1,1,0,0,1,1],
        #     [1,1,0,0,1,1],
        #     [1,1,0,0,1,1],
        #     [1,0,0,0,0,1],
        #     [0,1,1,1,1,0]
        # ], dtype=np.float32)
        # templates.append(t2)

    elif digit == 1:

        # t1 = np.array([
        #     [0,0,1,1,0,0],
        #     [0,1,1,1,0,0],
        #     [1,1,1,1,0,0],
        #     [0,0,1,1,0,0],
        #     [0,0,1,1,0,0],
        #     [0,0,1,1,0,0],
        #     [0,0,1,1,0,0],
        #     [0,0,1,1,0,0],
        #     [0,0,1,1,0,0],
        #     [0,0,1,1,0,0],
        #     [0,0,1,1,0,0],
        #     [1,1,1,1,1,1]
        # ], dtype=np.float32)
        # templates.append(t1)

        t2 = np.array([  
            [0,0,1,1,0,0],
            [0,0,1,1,0,0],
            [0,0,1,1,0,0],
            [0,0,1,1,0,0],
            [0,0,1,1,0,0],
            [0,0,1,1,0,0],
            [0,0,1,1,0,0],
            [0,0,1,1,0,0],
            [0,0,1,1,0,0],
            [0,0,1,1,0,0],
            [0,0,1,1,0,0],
            [0,0,1,1,0,0]
        ], dtype=np.float32)
        templates.append(t2)

    elif digit == 2:

        # t1 = np.array([
        #     [0,1,1,1,1,0],
        #     [1,1,0,0,1,1],
        #     [0,0,0,0,1,1],
        #     [0,0,0,1,1,0],
        #     [0,0,1,1,0,0],
        #     [0,1,1,0,0,0],
        #     [1,1,0,0,0,0],
        #     [1,1,0,0,0,0],
        #     [1,1,0,0,0,0],
        #     [1,1,0,0,1,1],
        #     [1,1,1,1,1,1],
        #     [1,1,1,1,1,1]
        # ], dtype=np.float32)
        # templates.append(t1)

        t2 = np.array([  
            [1,1,1,1,1,1],
            [1,1,1,1,1,1],
            [0,0,0,0,1,1],
            [0,0,0,0,1,1],
            [0,0,0,0,1,1],
            [0,0,0,0,1,1],
            [1,1,1,1,1,1],
            [1,1,1,1,1,1],
            [1,1,0,0,0,0],
            [1,1,0,0,0,0],
            [1,1,1,1,1,1],
            [1,1,1,1,1,1]
        ], dtype=np.float32)
        templates.append(t2)

    elif digit == 3:

        # t1 = np.array([
        #     [0,1,1,1,1,0],
        #     [1,1,0,0,1,1],
        #     [0,0,0,0,1,1],
        #     [0,0,0,0,1,1],
        #     [0,0,1,1,1,0],
        #     [0,0,0,0,1,1],
        #     [0,0,0,0,1,1],
        #     [0,0,0,0,1,1],
        #     [0,0,0,0,1,1],
        #     [1,1,0,0,1,1],
        #     [1,1,0,0,1,1],
        #     [0,1,1,1,1,0]
        # ], dtype=np.float32)
        # templates.append(t1)

        t2 = np.array([  
            [1,1,1,1,1,1],
            [1,1,1,1,1,1],
            [0,0,0,0,1,1],
            [0,0,0,0,1,1],
            [0,0,0,0,1,1],
            [1,1,1,1,1,1],
            [1,1,1,1,1,1],
            [0,0,0,0,1,1],
            [0,0,0,0,1,1],
            [0,0,0,0,1,1],
            [1,1,1,1,1,1],
            [1,1,1,1,1,1]
        ], dtype=np.float32)
        templates.append(t2)

    elif digit == 4:

        t1 = np.array([  
            [1,1,0,0,1,1],
            [1,1,0,0,1,1],
            [1,1,0,0,1,1],
            [1,1,0,0,1,1],
            [1,1,0,0,1,1],
            [1,1,1,1,1,1],
            [1,1,1,1,1,1],
            [0,0,0,0,1,1],
            [0,0,0,0,1,1],
            [0,0,0,0,1,1],
            [0,0,0,0,1,1],
            [0,0,0,0,1,1]
        ], dtype=np.float32)
        templates.append(t1)

        # t2 = np.array([
        #     [0,0,0,1,1,1],
        #     [0,0,1,1,1,1],
        #     [0,1,1,0,1,1],
        #     [1,1,0,0,1,1],
        #     [1,0,0,0,1,1],
        #     [1,0,0,0,1,1],
        #     [1,1,1,1,1,1],
        #     [1,1,1,1,1,1],
        #     [0,0,0,0,1,1],
        #     [0,0,0,0,1,1],
        #     [0,0,0,0,1,1],
        #     [0,0,0,0,1,1]
        # ], dtype=np.float32)
        # templates.append(t2)

    elif digit == 5:

        # t1 = np.array([
        #     [1,1,1,1,1,1],
        #     [1,1,0,0,0,0],
        #     [1,1,0,0,0,0],
        #     [1,1,1,1,1,0],
        #     [0,0,0,0,1,1],
        #     [0,0,0,0,1,1],
        #     [0,0,0,0,1,1],
        #     [0,0,0,0,1,1],
        #     [0,0,0,0,1,1],
        #     [1,1,0,0,1,1],
        #     [1,1,0,0,1,1],
        #     [0,1,1,1,1,0]
        # ], dtype=np.float32)
        # templates.append(t1)

        t2 = np.array([  
            [1,1,1,1,1,1],
            [1,1,1,1,1,1],
            [1,1,0,0,0,0],
            [1,1,0,0,0,0],
            [1,1,0,0,0,0],
            [1,1,1,1,1,1],
            [1,1,1,1,1,1],
            [0,0,0,0,1,1],
            [0,0,0,0,1,1],
            [0,0,0,0,1,1],
            [1,1,1,1,1,1],
            [1,1,1,1,1,1]
        ], dtype=np.float32)
        templates.append(t2)

    elif digit == 6:

        # t1 = np.array([
        #     [0,1,1,1,1,0],
        #     [1,1,0,0,1,1],
        #     [1,1,0,0,0,0],
        #     [1,1,0,0,0,0],
        #     [1,1,1,1,1,0],
        #     [1,1,0,0,1,1],
        #     [1,1,0,0,1,1],
        #     [1,1,0,0,1,1],
        #     [1,1,0,0,1,1],
        #     [1,1,0,0,1,1],
        #     [1,1,0,0,1,1],
        #     [0,1,1,1,1,0]
        # ], dtype=np.float32)
        # templates.append(t1)

        t2 = np.array([  
            [1,1,1,1,1,1],
            [1,1,1,1,1,1],
            [1,1,0,0,0,0],
            [1,1,0,0,0,0],
            [1,1,0,0,0,0],
            [1,1,1,1,1,1],
            [1,1,1,1,1,1],
            [1,1,0,0,1,1],
            [1,1,0,0,1,1],
            [1,1,0,0,1,1],
            [1,1,1,1,1,1],
            [1,1,1,1,1,1]
        ], dtype=np.float32)
        templates.append(t2)

    elif digit == 7:

        # t1 = np.array([
        #     [1,1,1,1,1,1],
        #     [0,0,0,0,1,1],
        #     [0,0,0,1,1,0],
        #     [0,0,1,1,0,0],
        #     [0,0,1,1,0,0],
        #     [0,1,1,0,0,0],
        #     [0,1,1,0,0,0],
        #     [1,1,0,0,0,0],
        #     [1,1,0,0,0,0],
        #     [1,1,0,0,0,0],
        #     [1,1,0,0,0,0],
        #     [1,1,0,0,0,0]
        # ], dtype=np.float32)
        # templates.append(t1)

        t2 = np.array([  
            [1,1,1,1,1,1],
            [1,1,1,1,1,1],
            [0,0,0,0,1,1],
            [0,0,0,0,1,1],
            [0,0,0,0,1,1],
            [0,0,0,0,1,1],
            [0,0,0,0,1,1],
            [0,0,0,0,1,1],
            [0,0,0,0,1,1],
            [0,0,0,0,1,1],
            [0,0,0,0,1,1],
            [0,0,0,0,1,1]
        ], dtype=np.float32)
        templates.append(t2)

    elif digit == 8:

        # t1 = np.array([
        #     [0,1,1,1,1,0],
        #     [1,1,0,0,1,1],
        #     [1,1,0,0,1,1],
        #     [1,1,0,0,1,1],
        #     [0,1,1,1,1,0],
        #     [1,1,0,0,1,1],
        #     [1,1,0,0,1,1],
        #     [1,1,0,0,1,1],
        #     [1,1,0,0,1,1],
        #     [1,1,0,0,1,1],
        #     [1,1,0,0,1,1],
        #     [0,1,1,1,1,0]
        # ], dtype=np.float32)
        # templates.append(t1)

        t2 = np.array([  
            [1,1,1,1,1,1],
            [1,1,1,1,1,1],
            [1,1,0,0,1,1],
            [1,1,0,0,1,1],
            [1,1,0,0,1,1],
            [1,1,1,1,1,1],
            [1,1,1,1,1,1],
            [1,1,0,0,1,1],
            [1,1,0,0,1,1],
            [1,1,0,0,1,1],
            [1,1,1,1,1,1],
            [1,1,1,1,1,1]
        ], dtype=np.float32)
        templates.append(t2)

    elif digit == 9:

        # t1 = np.array([
        #     [0,1,1,1,1,0],
        #     [1,1,0,0,1,1],
        #     [1,1,0,0,1,1],
        #     [1,1,0,0,1,1],
        #     [1,1,0,0,1,1],
        #     [0,1,1,1,1,1],
        #     [0,0,0,0,1,1],
        #     [0,0,0,0,1,1],
        #     [0,0,0,1,1,0],
        #     [0,0,1,1,0,0],
        #     [0,1,1,0,0,0],
        #     [1,1,0,0,0,0]
        # ], dtype=np.float32)
        # templates.append(t1)

        t2 = np.array([  
            [1,1,1,1,1,1],
            [1,1,0,0,1,1],
            [1,1,0,0,1,1],
            [1,1,0,0,1,1],
            [1,1,0,0,1,1],
            [1,1,1,1,1,1],
            [1,1,1,1,1,1],
            [0,0,0,0,1,1],
            [0,0,0,0,1,1],
            [0,0,0,0,1,1],
            [1,1,1,1,1,1],
            [1,1,1,1,1,1]
        ], dtype=np.float32)
        templates.append(t2)
    return templates


def generate_dataset_for_digit_with_noise_levels(digit, noise_levels, n_samples_per_template=50):
    
    templates = get_templates_for_digit(digit)
    dataset = []

    for noise in noise_levels:
        for template in templates:
            for _ in range(n_samples_per_template):
                sample = template.copy()
                # Apply noise: flip cells with probability = noise
                noise_mask = np.random.rand(12, 6) < noise
                sample[noise_mask] = 1 - sample[noise_mask]

                dataset.append(sample.flatten())


    return np.array(dataset, dtype=np.int64)
        


def generate_and_save_dataset(output_file, noise_levels, samples_per_template=500, is_training=True):
    
    all_data = []
    
    for digit in range(10):
        # Generate data for current digit
        data = generate_dataset_for_digit_with_noise_levels(
            digit, 
            noise_levels, 
            n_samples_per_template=samples_per_template
        )
        
        # Create DataFrame for current digit
        feature_columns = [f"pixel_{i}" for i in range(72)]
        df_data = pd.DataFrame(data, columns=feature_columns)
        df_data["label"] = digit
        
        all_data.append(df_data)
        print(f"Generated dataset for digit {digit}")
    
    # Combine all digits' data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Shuffle the dataset
    shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    shuffled_df.to_csv(output_file, index=False)
    print(f"Saved {'training' if is_training else 'testing'} dataset to {output_file}")


def main():
    # Define the noise levels
    # noise_levels = [0.005, 0.01, 0.02]
    noise_levels = [0,0]
    # Generate training data
    generate_and_save_dataset(
        # output_file="generated_data/gen_train_data.csv",
        output_file="new_data/gen_train_data.csv",
        noise_levels=noise_levels,
        samples_per_template=250,
        is_training=True
    )
    
    # Generate testing data
    generate_and_save_dataset(
        # output_file="generated_data/gen_test_data.csv",
        output_file="new_data/gen_test_data.csv",
        noise_levels=noise_levels,
        samples_per_template=50,
        is_training=False
    )

if __name__ == "__main__":
    main()