
import pandas as pd
import glob
import os
import numpy as np


#Combine datasets:
data_dir = '../datasets/classifier_2'
out_dir = '/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/datasets/classifier_v2_traintestdev'

# List to hold all cleaned DataFrames
all_dfs = []

# Loop through all files in the directory
for filename in os.listdir(data_dir):
    if filename.endswith('_train_dataset_classifier.csv'):
        filepath = os.path.join(data_dir, filename)

        # Read the CSV file
        df = pd.read_csv(filepath)

        # Remove rows with duplicate sequence + num_hits
        df = df.drop_duplicates(subset=['sequence', 'num_hits', 'accession'])

        # Extract species name from filename
        species = filename.replace('_train_dataset_classifier.csv', '')

        # Add species column
        df['species'] = species

        # Append to list
        all_dfs.append(df)

# Concatenate all DataFrames into one
combined_df = pd.concat(all_dfs, ignore_index=True)


#Get rid of invalid sequences
combined_df = combined_df[combined_df['sequence'].str.upper().str.fullmatch(r'[ACGT]+')]


#Sample to remove ~half of resistant rows while stratifying for accession without getting rid of any accessions entirely
# Filter resistant rows
resistant_df = combined_df[combined_df['phenotype'] == 'Resistant']
non_resistant_df = combined_df[combined_df['phenotype'] != 'Resistant']

# Desired drop rate
drop_rate = 0.5

# Group by accession
grouped = resistant_df.groupby('accession')

rows_to_drop = []

for accession, group in grouped:
    n = len(group)
    if n > 3:
        drop_n = int(np.floor(n * drop_rate))
        # Always leave at least one
        drop_n = min(drop_n, n - 1)
        rows_to_drop.extend(group.sample(drop_n, random_state=42).index)

# Drop the selected rows
resistant_df = resistant_df.drop(index=rows_to_drop)

# Recombine
balanced_df = pd.concat([resistant_df, non_resistant_df], ignore_index=True)



#Functions for data augmentation:
#double susceptible and quadruple Intermediate
complement_map = str.maketrans("ACGTacgt", "TGCAtgca")

def complement(seq):
    return seq.translate(complement_map)

def reverse_complement(seq):
    return complement(seq)[::-1]


#Run data augmentation:
augmented_rows = []
df = balanced_df

for idx, row in df.iterrows():
    pheno = row['phenotype']
    if pheno == 'Susceptible':
        # Add reverse complement
        new_row = row.copy()
        new_row['sequence'] = reverse_complement(row['sequence'])
        augmented_rows.append(new_row)
    elif pheno == 'Intermediate':
        # Add complement
        new_row_comp = row.copy()
        new_row_comp['sequence'] = complement(row['sequence'])
        augmented_rows.append(new_row_comp)
        
        # Add reverse complement
        new_row_revcomp = row.copy()
        new_row_revcomp['sequence'] = reverse_complement(row['sequence'])
        augmented_rows.append(new_row_revcomp)

# Append all new rows to the original DataFrame
df_augmented = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)

#Rearrange columns:
cols = [col for col in df_augmented.columns if col != 'phenotype'] + ['phenotype']
df_augmented = df_augmented[cols]


#Train/test split keeping accessions constrained to set:
#**maintain accessions within each group so consensus classifier uses test accessions only
from sklearn.model_selection import train_test_split

unique_accessions = df_augmented.drop_duplicates(subset='accession')

train_acc, temp_acc = train_test_split(
    unique_accessions,
    test_size=0.20,
    stratify=unique_accessions['phenotype'],
    random_state=42
)

dev_acc, test_acc = train_test_split(
    temp_acc,
    test_size=0.50,  # 50% of 20% = 10%
    stratify=temp_acc['phenotype'],
    random_state=42
)

train_df = df_augmented[df_augmented['accession'].isin(train_acc['accession'])]
test_df = df_augmented[df_augmented['accession'].isin(test_acc['accession'])]
dev_df = df_augmented[df_augmented['accession'].isin(dev_acc['accession'])]

train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
test_df  = test_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
dev_df   = dev_df.sample(frac=1.0, random_state=42).reset_index(drop=True)


#Output to csv:
train_df.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
test_df.to_csv(os.path.join(out_dir, 'test.csv'), index=False)
dev_df.to_csv(os.path.join(out_dir, 'dev.csv'), index=False)   


#Print class weights for training:
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

label_mapping = {'Resistant': 0, 'Intermediate': 1, 'Susceptible': 2}
train_df['label'] = train_df['phenotype'].map(label_mapping)

# Compute weights using manual labels
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1, 2]),
    y=train_df['label']
)

print(class_weights)
