#!/bin/bash
#SBATCH -p a100-long
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem 250G
#SBATCH --output res.txt

conda activate dna

export DATA_PATH=/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/finetune_data
export MAX_LENGTH=125
export LR=3e-5

# Training use DataParallel
cd DNABERT_2/finetune
python train.py \
    --model_name_or_path /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/bacteria_model \
    --data_path  ${DATA_PATH} \
    --kmer -1 \
    --run_name DNABERT2_${DATA_PATH} \
    --model_max_length ${MAX_LENGTH} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate ${LR} \
    --num_train_epochs 5 \
    --save_steps 200 \
    --output_dir output/dnabert2 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --warmup_steps 50 \
    --logging_steps 100 \
    --overwrite_output_dir True \
    --log_level info \
    --find_unused_parameters False \
#   --fp16
