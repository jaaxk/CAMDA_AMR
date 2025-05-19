#!/bin/bash
#SBATCH -p long-96core
#SBATCH --output res.txt

conda activate bioinfo

export INPUT_DIR=/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/de_novo_assembly/contigs_by_species
export NUM_SEQS_PER_ACC=132 #to match size of dbgwas augmentation set (v10)
export SUBSEQ_LEN=1000
export METADATA_CSV=/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/metadata/train_metadata.csv
export OUTPUT_CSV=/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/no_dbgwas/datasets/no_dbgwas_1000bp_132peracc.csv

python extract_random_subseqs.py --input_dir $INPUT_DIR --num_seqs_per_acc $NUM_SEQS_PER_ACC --subseq_len $SUBSEQ_LEN --metadata_csv $METADATA_CSV --output_csv $OUTPUT_CSV
