import pandas as pd

# Species mapping
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

file = '/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/get_final_sets/final_sets/thresh_5e-2_blast_80_pv3/dev.csv'

def convert_species():
    df = pd.read_csv(file)
    
    # Convert species to integers
    df['species'] = df['species'].map(species_mapping)
    
    # Save the updated dataframe
    output_file = file
    df.to_csv(output_file, index=False)
    print(f"Saved updated file to: {output_file}")

if __name__ == "__main__":
    convert_species()
