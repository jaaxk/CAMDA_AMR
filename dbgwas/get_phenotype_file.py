import pandas as pd
import os

metadata_path = '/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/metadata/train_metadata.csv'
contigs_path = '/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/de_novo_assembly/contigs_by_species/klebsiella_pneumoniae/'

metadata_df = pd.read_csv(metadata_path)

dbgwas_phenotypes = metadata_df[metadata_df['genus_species'] == 'Klebsiella pneumoniae'][['accession', 'phenotype']].drop_duplicates(subset='accession', keep='first')

accessions_in_dir = {
    filename.split('.')[0]
    for filename in os.listdir(contigs_path)
    if filename.endswith('.fasta')
}
print(f'accessions in directory: {len(accessions_in_dir)}')
dbgwas_phenotypes = dbgwas_phenotypes[dbgwas_phenotypes['accession'].isin(accessions_in_dir)]

dbgwas_phenotypes['phenotype'], unique_vals = pd.factorize(dbgwas_phenotypes['phenotype'])
dbgwas_phenotypes['Path'] = dbgwas_phenotypes['accession'].apply(lambda x: contigs_path+x+'.fasta')
dbgwas_phenotypes = dbgwas_phenotypes.rename(columns={'accession': 'ID', 'phenotype': 'Phenotype'})
dbgwas_phenotypes.to_csv('klebsiella_phenotypes.tsv', index=False, sep='\t')
