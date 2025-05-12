#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import shutil
#from tqdm import tqdm

# Default species list
DEFAULT_SPECIES = [
    'neisseria_gonorrhoeae', #done
    'staphylococcus_aureus', # - wont finish
    'streptococcus_pneumoniae', #done
    'salmonella_enterica', # - wont finish
    'klebsiella_pneumoniae', #done
    'escherichia_coli', #in progress
    'pseudomonas_aeruginosa', #in progress 
    'acinetobacter_baumannii', # - wont finish
    'campylobacter_jejuni' #done
]

class AMRPipeline:
    def __init__(self, metadata_path, base_dir, threads=96, species=None, subseq_length=1000, min_seqs=None, blast_identity=None, dbgwas_sig_level=None, contigs_dir=None, test_set=False):
        self.metadata_path = metadata_path
        self.base_dir = Path(base_dir)
        self.threads = threads
        self.species = species or DEFAULT_SPECIES
        self.metadata_df = pd.read_csv(metadata_path)
        self.subseq_length = subseq_length
        self.min_seqs = min_seqs
        self.blast_identity = blast_identity
        self.dbgwas_sig_level = dbgwas_sig_level
        self.contigs_dir = Path(contigs_dir)
        self.test_set = test_set

        self.metadata_df['genus_species'] = self.metadata_df['genus'].str.lower() + '_' + self.metadata_df['species'].str.lower()

        print(f"\nPipeline initialized for species: {self.species}")
        if self.test_set:
            print("Running on TEST set (no ground truth)")
        print(f"Metadata file: {self.metadata_path}")
        print(f"Base directory: {self.base_dir}")
        print(f"Number of threads: {self.threads}")
        print(f"Subsequence length: {self.subseq_length}")
        print(f"Minimum sequences: {self.min_seqs}")
        print(f"BLAST identity: {self.blast_identity}")
        print(f"DBGWAS p-value cutoff: {self.dbgwas_sig_level}")

    def check_prerequisites(self, species):
        """Check if all required files and directories exist for a species"""
        dir_name = species.lower().replace(' ', '_')
        contigs_path = self.contigs_dir / dir_name

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        if not contigs_path.exists():
            raise FileNotFoundError(f"Contigs directory not found for {species}: {contigs_path}")

        # Check if there are any FASTA files in the contigs directory
        fasta_files = list(contigs_path.glob('*.fasta'))
        if not fasta_files:
            raise FileNotFoundError(f"No FASTA files found in {contigs_path}")

        return str(contigs_path)

    def run_dbgwas(self, species, dir_name, contigs_path):
        """Run DBGWAS for a species
        
        This is only done for training, in test, use the same dbgwas output from training"""
        print(f"\n🔍 Running DBGWAS for {species}")

        # Get phenotypes for the species
        dbgwas_phenotypes = self.metadata_df[
            self.metadata_df['genus_species'] == species
        ][['accession', 'phenotype']].drop_duplicates(subset='accession', keep='first')

        # Get available assemblies
        accessions_in_dir = {
            filename.split('.')[0]
            for filename in os.listdir(contigs_path)
            if filename.endswith('.fasta')
        }
        print(f'Found {len(accessions_in_dir)} assemblies for {species}')

        # Filter phenotypes to only include available assemblies
        dbgwas_phenotypes = dbgwas_phenotypes[
            dbgwas_phenotypes['accession'].isin(accessions_in_dir)
        ]


        if dbgwas_phenotypes.empty:
            raise ValueError(f"No matching assemblies found for {species} in metadata")

        # Prepare phenotype file for DBGWAS
        label_mapping = {'Resistant': 0, 'Intermediate': 0, 'Susceptible': 1}
        dbgwas_phenotypes['phenotype'] = dbgwas_phenotypes['phenotype'].map(label_mapping)
        dbgwas_phenotypes['Path'] = dbgwas_phenotypes['accession'].apply(
            lambda x: os.path.join(contigs_path, f"{x}.fasta")
        )
        dbgwas_phenotypes = dbgwas_phenotypes.rename(
            columns={'accession': 'ID', 'phenotype': 'Phenotype'}
        )

        # Create output directory and save phenotype file
        pheno_file = f'{self.base_dir}/dbgwas/{dir_name}/{dir_name}_phenotypes.tsv'
        dbgwas_phenotypes.to_csv(pheno_file, index=False, sep='\t')

        print(f'Running DBGWAS on {len(dbgwas_phenotypes)} assemblies for {species}')

        # Run DBGWAS
        if not os.path.exists(f'{self.base_dir}/dbgwas/{dir_name}/dbgwas_output/step2'):
            print(f'No step 2 found, starting from DBGWAS from scratch')
            shutil.rmtree(f'{self.base_dir}/dbgwas/{dir_name}/dbgwas_output')
            cmd = (f'singularity run --bind {self.base_dir}:{self.base_dir} '
                f'docker://leandroishilima/dbgwas:0.5.4 '
                f'-nb-cores {self.threads} '
                f'-strains {pheno_file} '
                f'-output {self.base_dir}/dbgwas/{dir_name}/dbgwas_output')

        elif os.path.exists(f'{self.base_dir}/dbgwas/{dir_name}/dbgwas_output/step2') and not os.path.exists(f'{self.base_dir}/dbgwas/{dir_name}/dbgwas_output/step3'):
            print(f'No step 3 found, starting from DBGWAS step 2')
            shutil.rmtree(f'{self.base_dir}/dbgwas/{dir_name}/dbgwas_output/step2')
            cmd = (f'singularity run --bind {self.base_dir}:{self.base_dir} '
                f'docker://leandroishilima/dbgwas:0.5.4 '
                f'-nb-cores {self.threads} '
                f'-strains {pheno_file} '
                f'-output {self.base_dir}/dbgwas/{dir_name}/dbgwas_output'
                f'-skip1')

        elif os.path.exists(f'{self.base_dir}/dbgwas/{dir_name}/dbgwas_output/step3'):
            print('No final output foumd, but step 3 was found, starting from step 3')
            shutil.rmtree(f'{self.base_dir}/dbgwas/{dir_name}/dbgwas_output/step3')
            cmd = (f'singularity run --bind {self.base_dir}:{self.base_dir} '
                f'docker://leandroishilima/dbgwas:0.5.4 '
                f'-nb-cores {self.threads} '
                f'-strains {pheno_file} '
                f'-output {self.base_dir}/dbgwas/{dir_name}/dbgwas_output'
                f'-skip1 -skip2')

            
            
        if os.system(cmd) != 0:
            raise RuntimeError(f"DBGWAS failed for {species}")

    def get_significant_sequences(self, species, dir_name):
        """Extract significant sequences from DBGWAS output"""
        print(f"\n📊 Extracting significant sequences for {species}")

        dbgwas_output_file = os.path.join(self.base_dir, 'dbgwas', dir_name, 'dbgwas_output', 'textualOutput', 'all_comps_nodes_info.tsv')
        if not os.path.exists(dbgwas_output_file):
            raise FileNotFoundError(f"DBGWAS output not found: {dbgwas_output_file}")

        # Read DBGWAS output and filter significant sequences
        dbgwas_output = pd.read_csv(dbgwas_output_file, sep='\t', index_col=False)
        seqs = dbgwas_output[dbgwas_output['p-value'] < self.dbgwas_sig_level]['Sequence'].tolist()
        
        if not seqs:
            print(f"Warning: No significant sequences found for {species}") #should make empty list 'seqs'

        # Write sequences to FASTA file
        output_file = os.path.join(self.base_dir, 'dbgwas', dir_name, f'{dir_name}_sig_sequences.fasta')
        with open(output_file, 'w') as f:
            for i, seq in enumerate(seqs):
                seq_name = f'seq_{i}'
                f.write(f">{seq_name}\n{seq}\n")

        print(f"Saved {len(seqs)} significant sequences")

    def run_blast(self, species, dir_name, contigs_path):
        """Run BLAST for significant sequences against assemblies"""
        print(f"\n🧬 Running BLAST for {species}")

        db_dir = os.path.join(self.base_dir, 'dbgwas', dir_name, 'dbs')
        out_dir = os.path.join(self.base_dir, 'dbgwas', dir_name, 'blast_output')
        os.makedirs(db_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        sig_seqs = os.path.join(self.base_dir, 'dbgwas', dir_name, f'{dir_name}_sig_sequences.fasta')

        if not os.path.exists(sig_seqs):
            raise FileNotFoundError(f"Significant sequences file not found: {sig_seqs}")

        assemblies = [f for f in os.listdir(contigs_path) if f.endswith('.fasta')]

        #if not os.path.exists(db_dir):
        for assembly in assemblies:
            accession = assembly.split('.')[0]
            
            # Create BLAST database
            if not os.path.exists(os.path.join(db_dir, f'{accession}.nhr')):
                cmd = f'makeblastdb -in {os.path.join(contigs_path, assembly)} -dbtype nucl -out {db_dir}/{accession}'
                if os.system(cmd) != 0:
                    print(f"Warning: makeblastdb failed for {assembly}")
        #else:
        #print(f"BLAST databases already exist in {db_dir}, moving to BLAST step")
                        

        for assembly in assemblies:
            accession = assembly.split('.')[0]
            # Run BLAST
            cmd = (f'blastn -query {sig_seqs} -db {db_dir}/{accession} '
                  f'-max_target_seqs 1 -outfmt "6 qseqid sseqid pident length '
                  f'qstart qend sstart send sstrand bitscore" -perc_identity {self.blast_identity} '
                  f'-out {out_dir}/{accession}_hits.tsv')
            if os.system(cmd) != 0:
                print(f"Warning: BLAST failed for {assembly}")

    def get_train_dataset(self, species, dir_name, contigs_path):
        """Generate training dataset using get_train_dataset.py"""
        print(f"\n📚 Generating training dataset for {species}")

        if self.test_set:
            out_name = f'{dir_name}_test_dataset'
        else:
            out_name = f'{dir_name}_train_dataset'

        cmd = (f'python {self.base_dir}/dbgwas/scripts/get_train_dataset.py --camda_dataset {self.metadata_path} '
               f'--assembly_dir {contigs_path} '
               f'--blast_dir {os.path.join(self.base_dir, "dbgwas", dir_name, "blast_output")} '
               f'--output {os.path.join(self.base_dir, "dbgwas", dir_name, out_name)} '
               f"--species '{species}' "
               f'--subseq_length {self.subseq_length} '
               f'--min_seqs {self.min_seqs} '
               f'--test_set {self.test_set}')

        print(cmd)
        
        if os.system(cmd) != 0:
            raise RuntimeError(f"Failed to generate training dataset for {species}")
            

    def run_pipeline(self):
        """Run the complete pipeline for all specified species"""
        print(f"Starting AMR pipeline for {len(self.species)} species")
        
        # Check prerequisites for each species
        for species in self.species:
            print(f"\n{'='*80}\nProcessing {species}\n{'='*80}")
            
            
            # Setup
            dir_name = species.lower().replace(' ', '_')
            contigs_path = self.check_prerequisites(species)
            full_dir_name = os.path.join(str(self.base_dir), 'dbgwas', dir_name)
            if not os.path.exists(full_dir_name):
                os.makedirs(full_dir_name)

            # Run pipeline steps
            if not os.path.exists(f'{full_dir_name}/dbgwas_output/textualOutput/all_comps_nodes_info.tsv') and not self.test_set:
                self.run_dbgwas(species, dir_name, contigs_path)
            else:
                print(f" DBGWAS already completed for {species}")
            
            if not os.path.exists(f'{full_dir_name}/{dir_name}_sig_sequences.fasta'):
                self.get_significant_sequences(species, dir_name)
            else:
                print(f" Significant sequences already extracted for {species}")
            
            if not os.path.exists(f'{full_dir_name}/blast_output'):
                print(f'{full_dir_name}/blast_output does not exist')
                self.run_blast(species, dir_name, contigs_path)
            else:
                print(f" BLAST already completed for {species}")
            
            if not self.test_set:
                if not os.path.exists(f'{full_dir_name}/{dir_name}_train_dataset_classifier.csv'):
                    self.get_train_dataset(species, dir_name, contigs_path) # extract flanking sequences
                else:
                    print(f" Training dataset already extracted for {species}")
            else:
                if not os.path.exists(f'{full_dir_name}/{dir_name}_test_dataset_classifier.csv'):
                    self.get_train_dataset(species, dir_name, contigs_path) # extract flanking sequences
                else:
                    print(f" Test dataset already extracted for {species}")

            print(f"\n✅ Successfully completed pipeline for {species}")

            if not self.test_set:
                df = pd.read_csv(f'{full_dir_name}/{dir_name}_train_dataset_classifier.csv')
                num_unique_accs = len(df['accession'].unique())
                print(f" Number of unique accessions in training dataset: {num_unique_accs}")
                if num_unique_accs < 500:
                    print(f" WARNING: Not enough unique accessions in training dataset for {species} !!!!")
            else:
                df = pd.read_csv(f'{full_dir_name}/{dir_name}_test_dataset_classifier.csv')
                num_unique_accs = len(df['accession'].unique())
                print(f" Number of unique accessions in test dataset: {num_unique_accs}")
                if num_unique_accs < 500:
                    print(f" WARNING: Not enough unique accessions in test dataset for {species} !!!!")

                


            
            
            

def main():
    parser = argparse.ArgumentParser(description="AMR Pipeline for multiple species")
    parser.add_argument("--metadata", default="/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/metadata/training_dataset.csv",
                      help="Path to metadata CSV file (straight from CAMDA)")
    parser.add_argument("--base-dir", default="/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR",
                      help="Base directory for the project")
    parser.add_argument("--threads", type=int, default=96,
                      help="Number of threads to use for DBGWAS")
    parser.add_argument("--species", nargs="+",
                      help="List of species to process (default: all species)")
    parser.add_argument("--subseq_length", type=int, default=10000,
                      help="Length of subsequence to extract")
    parser.add_argument("--min_seqs", type=int,
                      help="Minimum number of sequences required for a species")
    parser.add_argument("--blast_identity", type=int,
                      help="BLAST identity threshold")
    parser.add_argument("--dbgwas_sig_level", type=float,
                      help="p-value cutoff for DBGWAS results")
    parser.add_argument("--contigs_dir", type=str,
                      help="Directory containing contigs", default="/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/de_novo_assembly/contigs_by_species")
    parser.add_argument("--test_set", action="store_true", default=False,
                      help="Run on for test set (no ground truth)")
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = AMRPipeline(
        metadata_path=args.metadata,
        base_dir=args.base_dir,
        threads=args.threads,
        species=args.species,
        subseq_length=args.subseq_length,
        min_seqs=args.min_seqs,
        blast_identity=args.blast_identity,
        dbgwas_sig_level=args.dbgwas_sig_level,
        contigs_dir=args.contigs_dir,
        test_set=args.test_set
    )
    
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()
