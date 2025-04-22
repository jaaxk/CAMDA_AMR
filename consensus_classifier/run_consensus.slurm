#!/bin/bash
#SBATCH -p long-40core
#SBATCH --output res.txt

python infer_accession_consensus.py --model_dir /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/outputs/example_checkpoint --input_csv /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/datasets/combined_classifier_augmented_no_accession_overlap/test.csv --output_csv output.csv
