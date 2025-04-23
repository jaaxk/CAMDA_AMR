#!/bin/bash
#SBATCH -p long-96core
#SBATCH --output res.txt

conda activate dna
export MODEL_DIR=/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/outputs/best-55pertrain
export TEST_CSV=/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/datasets/classifier_v5_55pertrainacc/test.csv
export OUTPUT_CSV=55_best.csv

python infer_accession_consensus.py --model_dir $MODEL_DIR --test_csv $TEST_CSV --output_csv $OUTPUT_CSV
