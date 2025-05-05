#!/bin/bash
#SBATCH -p long-40core
#SBATCH --time 48:00:00
#SBATCH --output res.txt

#find . -type d -name "dbs" -exec rm -r {} +
find . -type d -name "blast_output" -exec rm -r {} +
find . -type f -name "*sig_sequences.fasta" -exec rm {} +

python -u run_amr_pipeline.py --threads 40
