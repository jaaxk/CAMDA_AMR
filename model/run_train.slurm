#!/bin/bash
#SBATCH -p a100
#SBATCH --mem 250G
#SBATCH --output res.txt
#SBATCH --time 8:00:00

export HF_HOME=/gpfs/scratch/jvaska/HF_HOME
python train.py --data_path ../dbgwas/klebsiella_pneumoniae/klebsiella_pneumoniae_train_dataset_classifier.csv --output_dir ./klebsiella_pneumoniae
