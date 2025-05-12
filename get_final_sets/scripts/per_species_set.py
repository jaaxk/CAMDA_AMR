import pandas as pd
import sys
import shutil
import os

def per_species_set(input_dir, output_dir):
    train_df = pd.read_csv(f'{input_dir}/train.csv')
    dev_df = pd.read_csv(f'{input_dir}/dev.csv')
    test_df = pd.read_csv(f'{input_dir}/test.csv')

    species_mapping = {
    'klebsiella_pneumoniae': 0,
    'streptococcus_pneumoniae': 1,
    'escherichia_coli': 2,
    'campylobacter_jejuni': 3,
    'salmonella_enterica': 4,
    'neisseria_gonorrhoeae': 5,
    'staphylococcus_aureus': 6,
    'pseudomonas_aeruginosa': 7,
    'acinetobacter_baumannii': 8
    }

    for species, idx in species_mapping.items():
        train_df_species = train_df[train_df['species'] == idx]
        train_df_species = train_df_species.drop('species', axis=1)
        dev_df_species = dev_df[dev_df['species'] == idx]
        dev_df_species = dev_df_species.drop('species', axis=1)
        test_df_species = test_df[test_df['species'] == idx]
        test_df_species = test_df_species.drop('species', axis=1)
        os.makedirs(f'{output_dir}/{species}', exist_ok=True)
        train_df_species.to_csv(f'{output_dir}/{species}/train.csv', index=False)
        dev_df_species.to_csv(f'{output_dir}/{species}/dev.csv', index=False)
        test_df_species.to_csv(f'{output_dir}/{species}/test.csv', index=False)
        print(f"Saved {species} to {output_dir}/{species}")
        print(f'train length: {len(train_df_species)}')
        print(f'dev length: {len(dev_df_species)}')
        print(f'test length: {len(test_df_species)}')
        print(f'train pheno distribution: {train_df_species["phenotype"].value_counts()}')
        print(f'dev pheno distribution: {dev_df_species["phenotype"].value_counts()}')
        print(f'test pheno distribution: {test_df_species["phenotype"].value_counts()}')
        print(train_df_species.head())
        print()
        
    shutil.copyfile(f'{input_dir}/test.csv', f'{output_dir}/test.csv')
    

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python per_species_set.py <input_dir> <output_dir>")
        sys.exit(1)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    per_species_set(input_dir, output_dir)
    
