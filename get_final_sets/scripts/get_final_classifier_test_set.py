import pandas as pd
import os

FINAL_DATASET_PATH = os.environ['FINAL_DATASET_PATH']
DATA_DIR = os.environ['DATA_DIR']

#Combine datasets: this dir is cleared each iteration
data_dir = DATA_DIR

# List to hold all cleaned DataFrames
all_dfs = []

print('concatenating sets')
# Loop through all files in the directory
for filename in os.listdir(data_dir):
    if filename.endswith('_dataset_classifier.csv'):
        filepath = os.path.join(data_dir, filename)

        # Read the CSV file
        df = pd.read_csv(filepath)

        # Remove rows with duplicate sequence + num_hits
        #df = df.drop_duplicates(subset=['sequence', 'num_hits', 'accession'])

        # Append to list
        all_dfs.append(df)

# Concatenate all DataFrames into one
combined_df = pd.concat(all_dfs, ignore_index=True)
combined_df = combined_df[combined_df['sequence'].str.upper().str.fullmatch(r'[ACGT]+')]

combined_df.to_csv(f'{FINAL_DATASET_PATH}/final_test.csv', index=False)
