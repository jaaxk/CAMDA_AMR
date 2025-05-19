import pandas as pd
import sys
import shutil
import os

def per_antibiotic_set(input_dir, output_dir):
    train_df = pd.read_csv(f'{input_dir}/train.csv')
    dev_df = pd.read_csv(f'{input_dir}/dev.csv')
    test_df = pd.read_csv(f'{input_dir}/test.csv')

    antibiotic_mapping = {'GEN': 0, 
        'ERY': 1,
        'CAZ': 2,
        'TET': 3,
    }

    for antibiotic, idx in antibiotic_mapping.items():
        train_df_antibiotic = train_df[train_df['antibiotic'] == idx]
        train_df_antibiotic = train_df_antibiotic.drop('antibiotic', axis=1)
        dev_df_antibiotic = dev_df[dev_df['antibiotic'] == idx]
        dev_df_antibiotic = dev_df_antibiotic.drop('antibiotic', axis=1)
        test_df_antibiotic = test_df[test_df['antibiotic'] == idx]
        test_df_antibiotic = test_df_antibiotic.drop('antibiotic', axis=1)
        os.makedirs(f'{output_dir}/{antibiotic}', exist_ok=True)
        train_df_antibiotic.to_csv(f'{output_dir}/{antibiotic}/train.csv', index=False)
        dev_df_antibiotic.to_csv(f'{output_dir}/{antibiotic}/dev.csv', index=False)
        test_df_antibiotic.to_csv(f'{output_dir}/{antibiotic}/test.csv', index=False)
        print(f"Saved {antibiotic} to {output_dir}/{antibiotic}")
        print(f'train length: {len(train_df_antibiotic)}')
        print(f'dev length: {len(dev_df_antibiotic)}')
        print(f'test length: {len(test_df_antibiotic)}')
        #print(f'train pheno distribution: {train_df_antibiotic["phenotype"].value_counts()}')
        #print(f'dev pheno distribution: {dev_df_antibiotic["phenotype"].value_counts()}')
        #print(f'test pheno distribution: {test_df_antibiotic["phenotype"].value_counts()}')
        print(train_df_antibiotic.head())
        print()
        
    shutil.copyfile(f'{input_dir}/test.csv', f'{output_dir}/test.csv')
    

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python per_antibiotic_set.py <input_dir> <output_dir>")
        sys.exit(1)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    per_antibiotic_set(input_dir, output_dir)
    
