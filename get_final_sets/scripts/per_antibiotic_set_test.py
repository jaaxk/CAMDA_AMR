import pandas as pd
import sys
import shutil
import os

def per_antibiotic_set(input_dir, output_dir):
    df = pd.read_csv(os.path.join(input_dir, 'final_test.csv'))

    antibiotic_mapping = {'GEN': 0, 
        'ERY': 1,
        'CAZ': 2,
        'TET': 3,
    }

    for antibiotic, idx in antibiotic_mapping.items():
        df_antibiotic = df[df['antibiotic'] == idx]
        df_antibiotic = df_antibiotic.drop('antibiotic', axis=1)
        os.makedirs(f'{output_dir}/{antibiotic}', exist_ok=True)
        df_antibiotic.to_csv(f'{output_dir}/{antibiotic}/final_test.csv', index=False)
        print(f"Saved {antibiotic} to {output_dir}/{antibiotic}")
        print(f'dataset length: {len(df_antibiotic)}')
        print(df_antibiotic.head())
        print()
            

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python per_antibiotic_set.py <input_dir> <output_dir>")
        sys.exit(1)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    per_antibiotic_set(input_dir, output_dir)
    
