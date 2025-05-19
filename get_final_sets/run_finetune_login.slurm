#!/bin/bash
#SBATCH -p gpu-long
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem 125G
#SBATCH --output logs/res_ftlogin_%j.txt

source /gpfs/scratch/jvaska/miniconda3/etc/profile.d/conda.sh
conda activate dna

export DATA_PATH=$1
export RUN_NAME=$2
export MAX_LENGTH=250 #Changed to 250 for new 1000bp subseq dataset
export num_gpu=1
export MODEL_PATH=/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/bacteria_model
#export OMP_NUM_THREADS=7
export EVAL_AND_SAVE_STEPS=400 #This can be larger next hyperparam search to increase speed but lower it back to ~200 for final training
#export RES_WEIGHT=0.81626008
#export INT_WEIGHT=1.79831669
#export SUS_WEIGHT=0.82046254

export LR=3e-5
export OUT_DIR=/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/outputs/${RUN_NAME}
export WANDB_NAME=${RUN_NAME}
wandb login 0e16ac7c39d857e9bc3de95f06818dd4899bc8c1

echo "Running finetuning with dataset $DATA_PATH and run name $RUN_NAME"

torchrun --nproc_per_node=${num_gpu} /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/DNABERT_2/finetune/train.py \
--model_name_or_path ${MODEL_PATH} \
--data_path  ${DATA_PATH} \
--kmer -1 \
--run_name ${RUN_NAME} \
--model_max_length ${MAX_LENGTH} \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 16 \
--gradient_accumulation_steps 1 \
--learning_rate ${LR} \
--num_train_epochs 3 \
--fp16 \
--save_steps ${EVAL_AND_SAVE_STEPS} \
--output_dir ${OUT_DIR} \
--evaluation_strategy steps \
--eval_steps ${EVAL_AND_SAVE_STEPS} \
--warmup_steps 50 \
--logging_steps 100 \
--overwrite_output_dir True \
--log_level info \
--find_unused_parameters False \
#--res_weight ${RES_WEIGHT} \
#--int_weight ${INT_WEIGHT} \
#--sus_weight ${SUS_WEIGHT}

export BEST_MODEL_DIR=$OUT_DIR/best
echo "Running consensus evaluation with model $BEST_MODEL_DIR"
cp /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/model_files/* $BEST_MODEL_DIR
python /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/consensus_classifier/infer_accession_consensus.py --model_dir $BEST_MODEL_DIR --test_csv $DATA_PATH/test.csv --output_csv $OUT_DIR/accession_preds.csv
