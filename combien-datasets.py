import pandas as pd
import glob


csv_files = glob.glob("generated_data/gen_train_data.csv") + glob.glob("mnist_data/processed/processed_mnist_train.csv")
df_list = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)
shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
shuffled_df.to_csv("final_datasets/train_dataset.csv", index=False)
print("Shuffled and combined CSV saved as 'train_dataset.csv.csv'")

csv_files = glob.glob("generated_data/gen_test_data.csv") + glob.glob("mnist_data/processed/processed_mnist_test.csv")
df_list = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)
shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
shuffled_df.to_csv("final_datasets/test_dataset.csv", index=False)
print("Shuffled and combined CSV saved as 'test_dataset.csv.csv'")