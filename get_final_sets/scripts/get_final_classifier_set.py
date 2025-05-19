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

print('splitting into train, test, and dev')
# Get list of accessions for train, test, and dev
train_accs = set(line.strip() for line in open('/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/get_final_sets/train_test_dev_accs/train_accs.txt'))
dev_accs = set(line.strip() for line in open('/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/get_final_sets/train_test_dev_accs/dev_accs.txt'))
test_accs = set(line.strip() for line in open('/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/get_final_sets/train_test_dev_accs/test_accs.txt'))

# Split into train, test, and dev
train_df = combined_df[combined_df['accession'].isin(train_accs)]
dev_df = combined_df[combined_df['accession'].isin(dev_accs)]
test_df = combined_df[combined_df['accession'].isin(test_accs)]

# Save to file
#train_df.to_csv(f'{FINAL_DATASET_PATH}/train.csv', index=False)
dev_df.to_csv(f'{FINAL_DATASET_PATH}/dev.csv', index=False)
test_df.to_csv(f'{FINAL_DATASET_PATH}/test.csv', index=False)

#Best results were previously on augmentation of int 3x and sus 2x, so will use those
complement_map = str.maketrans("ACGTacgt", "TGCAtgca")
def complement(seq):
    return seq.translate(complement_map)
def reverse_complement(seq):
    return complement(seq)[::-1]

print('augmenting train set')
#Run data augmentation:
augmented_rows = []
df = train_df

for idx, row in df.iterrows():
    pheno = int(row['phenotype'])
    if pheno == 1:
        # Add reverse complement
        new_row = row.copy()
        new_row['sequence'] = reverse_complement(row['sequence'])
        augmented_rows.append(new_row)

# Append all new rows to the original DataFrame
df_augmented = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)

#If over 500k, ill get a memory error during finetuning
#if len(df_augmented) > 500000:
#    print(f'sampling train set to 500k from {len(df_augmented)}')
#    df_augmented = df_augmented.sample(n=500000, random_state=42)

print('saving train set')
#Save
train_df = df_augmented
train_df.to_csv(f'{FINAL_DATASET_PATH}/train.csv', index=False)
print('train set pheno counts: ')
print(train_df['phenotype'].value_counts())

    
