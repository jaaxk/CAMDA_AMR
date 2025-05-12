#!/bin/bash
#SBATCH -p hbm-long-96core
#SBATCH --time 48:00:00
#SBATCH --output res.txt

#find . -type d -name "dbs" -exec rm -r {} +
#find . -type d -name "blast_output" -exec rm -r {} +
#find . -type d -name "dbs" -exec rm -r {} +
#find . -type f -name "*sig_sequences.fasta" -exec rm {} +
#find . -type d -name "_classifier.csv" -exec rm {} +

python -u run_amr_pipeline.py --threads 96 --metadata /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/metadata/testing_template.csv --subseq_length 1000 --min_seqs 55 --blast_identity 80 --dbgwas_sig_level 5e-2 --contigs_dir /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/de_novo_assembly/test_assemblies/contigs_by_species/ --test_set
