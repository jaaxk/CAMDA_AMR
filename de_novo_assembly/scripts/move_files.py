import os
import shutil
import pandas as pd

# Load the metadata
metadata_path = '/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/metadata/test_metadata.csv'
output_base = '/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/de_novo_assembly/test_assemblies/contigs_by_species'
contigs_dir = '/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/de_novo_assembly/test_assemblies/contigs'

df = pd.read_csv(metadata_path, usecols=['accession', 'genus_species'])

# Remove duplicates, keeping the first occurrence
df = df.drop_duplicates(subset='accession', keep='first')

# Create a mapping from accession to genus_species
accession_to_species = dict(zip(df['accession'], df['genus_species']))

# Prepare output base path
os.makedirs(output_base, exist_ok=True)

# Get all file names in 'contigs' directory
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

# --- New logic: Check for missing accessions and move from fastp if needed ---
output_base = 'contigs_by_species'
#fastp_dir = os.path.join('..', 'data', 'train_data', 'fastp')

# Group metadata by species
species_to_accessions = df.groupby('genus_species')['accession'].apply(set).to_dict()

for genus_species, expected_accessions in species_to_accessions.items():
    species_dir = os.path.join(output_base, genus_species.lower().replace(' ', '_'))
    os.makedirs(species_dir, exist_ok=True)
    # Get accessions present in the directory (from filenames)
    present_accessions = set()
    for fname in os.listdir(species_dir):
        if fname.endswith('.fasta'):
            present_accessions.add(fname.split('.')[0])
    # Find missing accessions
    missing_accessions = expected_accessions - present_accessions
    print(f'Found {len(missing_accessions)} missing accessions for {genus_species}')
    for accession in missing_accessions:
        print(f'accession {accession} missing from {genus_species} directory')
        
        """# Search fastp_dir for a file containing the accession
        found = False
        if os.path.exists(fastp_dir):
            for fastp_file in os.listdir(fastp_dir):
                if accession in fastp_file:
                    src_path = os.path.join(fastp_dir, fastp_file)
                    dst_path = os.path.join(species_dir, f'{accession}.fasta')
                    shutil.move(src_path, dst_path)
                    print(f'Moved {fastp_file} to {dst_path}')
                    found = True
                    break
        if not found:
            print(f'Warning: Could not find file for missing accession {accession} in fastp directory.')"""

