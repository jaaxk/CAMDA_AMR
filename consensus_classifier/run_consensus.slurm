#!/bin/bash
#SBATCH -p long-96core
#SBATCH --output res_nomod.txt

conda activate dna
export MODEL_DIR=/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/outputs/mv3_pv3_cpu/checkpoint-26000
export TEST_CSV=/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/get_final_sets/final_sets/thresh_5e-2_blast_80_pv3/test.csv
export OUTPUT_CSV=outputs/mv3_pv3_5e-2

#export MODEL_DIR=/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/outputs/1000bp_nomod_classweights/best
#export TEST_CSV=/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/datasets/classifier_v8_100bp_nomod/test.csv
#export OUTPUT_CSV=outputs/1000bp_nomod_classweights.csv

python infer_accession_consensus.py --model_dir $MODEL_DIR --test_csv $TEST_CSV --output_csv $OUTPUT_CSV
