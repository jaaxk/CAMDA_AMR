#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
#from tqdm import tqdm

# Default species list
DEFAULT_SPECIES = [
    'Neisseria gonorrhoeae',
    'Staphylococcus aureus',
    'Streptococcus pneumoniae',
    'Salmonella enterica',
    'Klebsiella pneumoniae',
    'Escherichia coli',
    'Pseudomonas aeruginosa',
    'Acinetobacter baumannii',
    'Campylobacter jejuni'
]

class AMRPipeline:
    def __init__(self, metadata_path, base_dir, threads=96, species=None):
        self.metadata_path = metadata_path
        self.base_dir = Path(base_dir)
        self.threads = threads
        self.species = species or DEFAULT_SPECIES
        self.metadata_df = pd.read_csv(metadata_path)

    def check_prerequisites(self, species):
        """Check if all required files and directories exist for a species"""
        dir_name = species.lower().replace(' ', '_')
        contigs_path = self.base_dir / 'de_novo_assembly/contigs_by_species' / dir_name

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
        """Run DBGWAS for a species"""
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
        dbgwas_phenotypes['phenotype'], unique_vals = pd.factorize(dbgwas_phenotypes['phenotype'])
        dbgwas_phenotypes['Path'] = dbgwas_phenotypes['accession'].apply(
            lambda x: os.path.join(contigs_path, f"{x}.fasta")
        )
        dbgwas_phenotypes = dbgwas_phenotypes.rename(
            columns={'accession': 'ID', 'phenotype': 'Phenotype'}
        )

        # Create output directory and save phenotype file
        os.makedirs(dir_name, exist_ok=True)
        pheno_file = f'{dir_name}/{dir_name}_phenotypes.tsv'
        dbgwas_phenotypes.to_csv(pheno_file, index=False, sep='\t')

        # Run DBGWAS
        cmd = (f'singularity run --bind {self.base_dir}:{self.base_dir} '
               f'docker://leandroishilima/dbgwas:0.5.4 '
               f'-nb-cores {self.threads} '
               f'-strains {pheno_file} '
               f'-output {dir_name}/dbgwas_output')
        
        if os.system(cmd) != 0:
            raise RuntimeError(f"DBGWAS failed for {species}")

    def get_significant_sequences(self, species, dir_name):
        """Extract significant sequences from DBGWAS output"""
        print(f"\n📊 Extracting significant sequences for {species}")

        dbgwas_output_file = f'{dir_name}/dbgwas_output/textualOutput/all_comps_nodes_info.tsv'
        if not os.path.exists(dbgwas_output_file):
            raise FileNotFoundError(f"DBGWAS output not found: {dbgwas_output_file}")

        # Read DBGWAS output and filter significant sequences
        dbgwas_output = pd.read_csv(dbgwas_output_file, sep='\t', index_col=False)
        seqs = dbgwas_output[dbgwas_output['Significant?'] == 'Yes']['Sequence'].tolist()
        
        if not seqs:
            print(f"Warning: No significant sequences found for {species}")
            return

        # Write sequences to FASTA file
        output_file = f'{dir_name}/{dir_name}_sig_sequences.fasta'
        with open(output_file, 'w') as f:
            for i, seq in enumerate(seqs):
                seq_name = f'seq_{i}'
                f.write(f">{seq_name}\n{seq}\n")

        print(f"Saved {len(seqs)} significant sequences")

    def run_blast(self, species, dir_name, contigs_path):
        """Run BLAST for significant sequences against assemblies"""
        print(f"\n🧬 Running BLAST for {species}")

        db_dir = f'{dir_name}/dbs'
        out_dir = f'{dir_name}/blast_output'
        os.makedirs(db_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        sig_seqs = f'{dir_name}/{dir_name}_sig_sequences.fasta'

        if not os.path.exists(sig_seqs):
            raise FileNotFoundError(f"Significant sequences file not found: {sig_seqs}")

        assemblies = [f for f in os.listdir(contigs_path) if f.endswith('.fasta')]
        for assembly in assemblies:
            accession = assembly.split('.')[0]
            
            # Create BLAST database
            cmd = f'makeblastdb -in {os.path.join(contigs_path, assembly)} -dbtype nucl -out {db_dir}/{accession}'
            if os.system(cmd) != 0:
                print(f"Warning: makeblastdb failed for {assembly}")
                continue

            # Run BLAST
            cmd = (f'blastn -query {sig_seqs} -db {db_dir}/{accession} '
                  f'-max_target_seqs 1 -outfmt "6 qseqid sseqid pident length '
                  f'qstart qend sstart send sstrand bitscore" -perc_identity 85 '
                  f'-out {out_dir}/{accession}_hits.tsv')
            if os.system(cmd) != 0:
                print(f"Warning: BLAST failed for {assembly}")

    def get_train_dataset(self, species, dir_name, contigs_path):
        """Generate training dataset using get_train_dataset.py"""
        print(f"\n📚 Generating training dataset for {species}")

        cmd = (f'python {self.base_dir}/dbgwas/scripts/get_train_dataset.py --metadata {self.metadata_path} '
               f'--assembly_dir {contigs_path} '
               f'--blast_dir {dir_name}/blast_output '
               f'--output {dir_name}/{dir_name}_train_dataset '
               f"--species '{species}'")

        print(cmd)
        
        if os.system(cmd) != 0:
            raise RuntimeError(f"Failed to generate training dataset for {species}")

    def run_pipeline(self):
        """Run the complete pipeline for all specified species"""
        print(f"Starting AMR pipeline for {len(self.species)} species")
        
        # Check prerequisites for each species
        for species in self.species:
            print(f"\n{'='*80}\nProcessing {species}\n{'='*80}")
            
            try:
                # Setup
                dir_name = species.lower().replace(' ', '_')
                contigs_path = self.check_prerequisites(species)

                # Run pipeline steps
                if not os.path.exists(f'{self.base_dir}/dbgwas/{dir_name}/dbgwas_output'):
                    self.run_dbgwas(species, dir_name, contigs_path)
                else:
                    print(f" DBGWAS already completed for {species}")
                
                if not os.path.exists(f'{self.base_dir}/dbgwas/{dir_name}/{dir_name}_sig_sequences.fasta'):
                    self.get_significant_sequences(species, dir_name)
                else:
                    print(f" Significant sequences already extracted for {species}")
                
                if not os.path.exists(f'{self.base_dir}/dbgwas/{dir_name}/blast_output'):
                    self.run_blast(species, dir_name, contigs_path)
                else:
                    print(f" BLAST already completed for {species}")
                
                if not os.path.exists(f'{self.base_dir}/dbgwas/{dir_name}/{dir_name}_train_dataset_finetune.csv'):
                    self.get_train_dataset(species, dir_name, contigs_path) # extract flanking sequences
                else:
                    print(f" Training dataset already extracted for {species}")
                    
                print(f"\n✅ Successfully completed pipeline for {species}")
                
            except Exception as e:
                print(f"\n❌ Error processing {species}: {str(e)}")
                print("Continuing with next species...")

def main():
    parser = argparse.ArgumentParser(description="AMR Pipeline for multiple species")
    parser.add_argument("--metadata", default="/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/metadata/train_metadata.csv",
                      help="Path to metadata CSV file")
    parser.add_argument("--base-dir", default="/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR",
                      help="Base directory for the project")
    parser.add_argument("--threads", type=int, default=96,
                      help="Number of threads to use for DBGWAS")
    parser.add_argument("--species", nargs="+",
                      help="List of species to process (default: all species)")
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = AMRPipeline(
        metadata_path=args.metadata,
        base_dir=args.base_dir,
        threads=args.threads,
        species=args.species
    )
    
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()
