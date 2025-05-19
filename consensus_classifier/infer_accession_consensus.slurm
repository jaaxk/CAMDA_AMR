#!/bin/bash
#SBATCH -p hbm-long-96core


python infer_accession_consensus.py \
--model_dir /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/outputs/thresh_5e-2_blast_80_pv4/best_full \
--test_csv ../get_final_sets/final_sets/TRAIN_CLASSIFIER_NO_TOPUP/test.csv \
--run_name full_model \
--output_csv ./outputs/full_model
