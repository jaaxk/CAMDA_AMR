#!/bin/bash
#SBATCH -p a100-long
#SBATCH -t 6:00:00
#SBATCH --gres=gpu:4
#SBATCH --mail-type=BEGIN
##SBATCH --mail-user=jack.vaska@stonybrook.edu

python infer_mixed_models.py \
	--test_csv /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/get_final_sets/final_sets/test_2000_final/final_test.csv \
	--output_csv ./FINAL_TEST_2000.csv \
	--model_max_length 500 \
	--final_set \
	--test_template ../metadata/testing_template.csv
