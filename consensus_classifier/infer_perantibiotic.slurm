#!/bin/bash
#SBATCH -p hbm-long-96core
#SBATCH -t 24:00:00

python infer_perantibiotic.py \
	--models_dir ../finetune/outputs/thresh_5e-2_subseq_1000_pv4_PER_ANTIBIOTIC/ \
	--test_csv ../get_final_sets/final_sets/test_data_no_subsample/final_test.csv \
	--output_csv ./outputs/test_template_out_1.csv \
	--model_max_length 250 \
	--final_set \
	--test_template ../metadata/testing_template.csv	
