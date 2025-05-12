#!/bin/bash
#SBATCH -p hbm-long-96core
#SBATCH -t 12:00:00

export OMP_NUM_THREADS=96

python infer_perspecies.py --models_dir /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/outputs/1000_perspecies --test_csv /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/get_final_sets/final_sets/thresh_5e-2_subseq_1000_PER_SPECIES/test.csv --output_csv ./outputs/1000_perspecies.csv --model_max_length 250 --run_name 2species_fullmodel
