import os
import shutil
import pandas as pd

# Load the metadata
metadata_path = '../metadata/train_metadata.csv'
df = pd.read_csv(metadata_path, usecols=['accession', 'genus_species'])

# Remove duplicates, keeping the first occurrence
df = df.drop_duplicates(subset='accession', keep='first')

# Create a mapping from accession to genus_species
accession_to_species = dict(zip(df['accession'], df['genus_species']))

# Prepare output base path
output_base = 'contigs_by_species'
os.makedirs(output_base, exist_ok=True)

# Get all file names in 'contigs' directory
contigs_dir = 'contigs'
for filename in os.listdir(contigs_dir):
    if filename.endswith('.fasta'):
        accession = filename.split('.')[0]

        # Get the species for the accession
        genus_species = accession_to_species.get(accession).lower().replace(' ', '_')
        if genus_species:
            # Create species-specific directory if it doesn't exist
            species_dir = os.path.join(output_base, genus_species)
            os.makedirs(species_dir, exist_ok=True)

            # Move the file
            src_path = os.path.join(contigs_dir, filename)
            dst_path = os.path.join(species_dir, filename)
            shutil.move(src_path, dst_path)
        else:
            print(f'Warning: Accession {accession} not found in metadata.')

