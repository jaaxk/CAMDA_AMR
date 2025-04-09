import pandas as pd
import os

metadata_path = '/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/metadata/train_metadata.csv'
metadata_df = pd.read_csv(metadata_path)
threads = 96

"""species_list = ['Neisseria gonorrhoeae', 'Staphylococcus aureus',
       'Streptococcus pneumoniae', 'Salmonella enterica',
       'Klebsiella pneumoniae', 'Escherichia coli',
       'Pseudomonas aeruginosa', 'Acinetobacter baumannii',
       'Campylobacter jejuni']"""

species_list = ['Klebsiella pneumoniae']

def run_dbgwas(species, dir_name, contigs_path):

    dbgwas_phenotypes = metadata_df[metadata_df['genus_species'] == species][['accession', 'phenotype']].drop_duplicates(subset='accession', keep='first')

    accessions_in_dir = {
        filename.split('.')[0]
        for filename in os.listdir(contigs_path)
        if filename.endswith('.fasta')
    }
    print(f'accessions in {species} directory: {len(accessions_in_dir)}')
    dbgwas_phenotypes = dbgwas_phenotypes[dbgwas_phenotypes['accession'].isin(accessions_in_dir)]

    dbgwas_phenotypes['phenotype'], unique_vals = pd.factorize(dbgwas_phenotypes['phenotype'])
    dbgwas_phenotypes['Path'] = dbgwas_phenotypes['accession'].apply(lambda x: contigs_path+x+'.fasta')
    dbgwas_phenotypes = dbgwas_phenotypes.rename(columns={'accession': 'ID', 'phenotype': 'Phenotype'})

    os.makedirs(dir_name, exist_ok=True)
    dbgwas_phenotypes.to_csv(f'{dir_name}/{dir_name}_phenotypes.tsv', index=False, sep='\t')

    os.system(f'singularity run --bind /gpfs/scratch/jvaska/CAMDA_AMR:/gpfs/scratch/jvaska/CAMDA_AMR docker://leandroishilima/dbgwas:0.5.4 -nb-cores {threads} -strains {dir_name}/{dir_name}_phenotypes.tsv -output {dir_name}/dbgwas_output')

def get_sig_seqs(species, dir_name)

    dbgwas_output = pd.read_csv(f'{dir_name}/dbgwas_output/textualOutput/all_comps_nodes_info.tsv', sep='\t', index_col=False)
    seqs = dbgwas_output[dbgwas_output['Significant?'] == 'Yes']['Sequence'].tolist()
    with open(f'{dir_name}/{dir_name}_sig_sequences.fasta', 'w') as f:
        for i, seq in enumerate(seqs):
            seq_name = f'seq_{i}'
            f.write(f">{seq_name}\n")
            f.write(f"{seq}\n")

def run_blast(species, dir_name, contigs_path):

    db_dir = f'{dir_name}/dbs'
    out_dir = f'{dir_name}/blast_output'
    os.makedirs(db_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    sig_seqs = f'{dir_name}/{dir_name}_sig_sequences.fasta'

    for assembly in os.listdir(contigs_path):
        if not assembly.endswith('.fasta'):
            print(f'{assembly} not a fasta file')
            break

        accession = assembly.split('.')[0]
        os.system(f'makeblastdb -in {assembly} -dbtype nucl -out {db_dir}/{accession}')
        os.system(f"""blastn -query {sig_seqs} -db {db_dir}/{accession} -max_target_seqs 1 -outfmt "6 qseqid sseqid pident length qstart qend sstart send sstrand bitscore" -perc_identity 85 -out {out_dir}/{accession}_hits.tsv""")


def main():
    
    for species in species_list:
        dir_name = species.lower().replace(' ', '_')
        contigs_path = f'/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/de_novo_assembly/contigs_by_species/{dir_name}/'

       # print('Running DBGWAS')
       # run_dbgwas(species, dir_name, contigs_path)

        get_sig_seqs(species, dir_name)

        print('Running BLAST')
        run_blast(species, dir_name, contigs_path)

        print('Getting train dataset')
        os.system(f'python get_train_dataset.py --metadata {metadata_path} \
                  --assembly_dir {contigs_path} \
                  --blast_dir {dir_name}/blast_output --output {dir_name}/{dir_name}_train_dataset.csv')

if __name__ == '__main__':
    main()
