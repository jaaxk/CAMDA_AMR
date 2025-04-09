#!/bin/bash
#SBATCH -p long-96core
#SBATCH -t 48:00:00
#SBATCH -o res.txt

conda activate bioinfo
python get_train_datasets.py
