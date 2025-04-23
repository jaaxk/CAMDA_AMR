#!/bin/bash
#SBATCH -p long-96core
#SBATCH --time 48:00:00
#SBATCH --output res.txt

python run_amr_pipeline.py
