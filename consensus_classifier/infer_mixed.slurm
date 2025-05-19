#!/bin/bash
#SBATCH -p hbm-long-96core
#SBATCH -t 12:00:00

python infer_mixed_models.py \
	--test_csv /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/get_final_sets/final_sets/TRAIN_CLASSIFIER_NO_TOPUP/test.csv  \
	--output_csv ./outputs/FINAL_MIXED_NO_TOPUP_TEST \
	--model_max_length 250 \
	--run_name train_mixedmodels_notopup
