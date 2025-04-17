#!/bin/bash
#SBATCH -p a100-long
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem 250G
#SBATCH --output res.txt

conda activate dna

export DATA_PATH=/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/finetune_data
export MAX_LENGTH=125
#export LR=5e-5
export num_gpu=4
export MODEL_PATH=/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/bacteria_model
export OUT_DIR=/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/output
export OMP_NUM_THREADS=16

#wandb login 0e16ac7c39d857e9bc3de95f06818dd4899bc8c1

cd DNABERT_2/finetune
for LR in 1e-3 3e-3 1e-4 3e-4 1e-5 3e-5 1e-6 3e-6
do
     echo "Running with LR=$LR"
     export LR=$LR

     torchrun --nproc_per_node=${num_gpu} train.py \
     --model_name_or_path ${MODEL_PATH} \
     --data_path  ${DATA_PATH} \
     --kmer -1 \
     --run_name DNABERT2_${DATA_PATH} \
     --model_max_length ${MAX_LENGTH} \
     --per_device_train_batch_size 8 \
     --per_device_eval_batch_size 16 \
     --gradient_accumulation_steps 1 \
     --learning_rate ${LR} \
     --num_train_epochs 10 \
     --fp16 \
     --save_steps 200 \
     --output_dir ${OUT_DIR} \
     --evaluation_strategy steps \
     --eval_steps 200 \
     --warmup_steps 50 \
     --logging_steps 100 \
     --overwrite_output_dir True \
     --log_level info \
     --find_unused_parameters False

done

