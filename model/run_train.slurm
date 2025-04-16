#!/bin/bash
#SBATCH -p a100-long
#SBATCH --mem 250G
#SBATCH --output res.txt
#SBATCH --time 48:00:00
#SBATCH --gres=gpu:4

source amr_env/bin/activate
export HF_HOME=/gpfs/scratch/jvaska/HF_HOME
export OMP_NUM_THREADS=16
export DATA_PATH=/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/dbgwas/datasets/classifier/combined_classifier_train.csv
export MODEL_PATH=/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/output_2/best-checkpoint-6400
export OUT_DIR=/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/model/output

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/model/train.py \
    --data_path ${DATA_PATH} \
    --model_path ${MODEL_PATH} \
    --output_dir ${OUT_DIR} \
    --batch_size 32 \
    --epochs 20 \
    --num_workers 1
