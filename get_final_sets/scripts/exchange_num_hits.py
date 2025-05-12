import os
import sys
import pandas as pd
from pathlib import Path

def exchange_num_hits(from_dir, to_dir, output_dir):
    # List of dataset names to process
    datasets = ['train', 'test', 'dev']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset in datasets:
        # Construct file paths
        from_file = os.path.join(from_dir, f'{dataset}.csv')
        to_file = os.path.join(to_dir, f'{dataset}.csv')
        output_file = os.path.join(output_dir, f'{dataset}.csv')
        
        # Check if files exist
        if not os.path.exists(from_file):
            print(f"Error: Source file {from_file} not found")
            continue
        if not os.path.exists(to_file):
            print(f"Error: Target file {to_file} not found")
            continue
        
        try:
            # Read both CSV files
            from_df = pd.read_csv(from_file)
            to_df = pd.read_csv(to_file)
            
            # Create a mapping of accession to num_hits from the source file
            num_hits_map = from_df.set_index('accession')['num_hits'].to_dict()
            
            # Update num_hits in the target file, setting to 0 when accession is not found
            to_df['num_hits'] = to_df['accession'].map(num_hits_map).fillna(0)
            
            # Save the updated file to output directory
            to_df.to_csv(output_file, index=False)
            print(f"Successfully updated num_hits for {dataset} dataset and saved to {output_file}")
            
        except Exception as e:
            print(f"Error processing {dataset} dataset: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python exchange_num_hits.py <from_directory> <to_directory> <output_directory>")
        sys.exit(1)
    
    from_dir = sys.argv[1]
    to_dir = sys.argv[2]
    output_dir = sys.argv[3]
    
    if not os.path.isdir(from_dir):
        print(f"Error: {from_dir} is not a valid directory")
        sys.exit(1)
    if not os.path.isdir(to_dir):
        print(f"Error: {to_dir} is not a valid directory")
        sys.exit(1)
    
    exchange_num_hits(from_dir, to_dir, output_dir)