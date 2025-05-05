#!/bin/bash
#SBATCH -p a100-long
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem 250G
#SBATCH --output res1.txt

conda activate dna

#export DATA_PATH=/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/finetune_data
export DATA_PATH=/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/datasets/classifier_v14_nosigthreshold_dropdupes_augment
export MAX_LENGTH=250 #Changed to 250 for new 1000bp subseq dataset
#export LR=5e-5
export num_gpu=4
export MODEL_PATH=/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/bacteria_model
export OMP_NUM_THREADS=16
#export RES_WEIGHT=0.81626008
#export INT_WEIGHT=1.79831669
#export SUS_WEIGHT=0.82046254

wandb login 0e16ac7c39d857e9bc3de95f06818dd4899bc8c1

cd DNABERT_2/finetune


#for LR in 1e-3 3e-3 1e-4 3e-4 1e-5 3e-5 1e-6 3e-6; do

    #export LR=$LR
    #export WANDB_NAME=all_features_regressiom_lr_${LR}
    #export OUT_DIR=/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/outputs/output_all_features_regressiom_lr_${LR}

    export LR=3e-5
    export OUT_DIR=/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/outputs/v14_nosigthreshold_dropdupes_augment
    export WANDB_NAME=v14_nosigthreshold_dropdupes_augment

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
    --num_train_epochs 7 \
    --fp16 \
    --save_steps 200 \
    --output_dir ${OUT_DIR} \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --warmup_steps 50 \
    --logging_steps 100 \
    --overwrite_output_dir True \
    --log_level info \
    --find_unused_parameters False \
    #--res_weight ${RES_WEIGHT} \
    #--int_weight ${INT_WEIGHT} \
    #--sus_weight ${SUS_WEIGHT}

#done

export BEST_MODEL_DIR=$OUT_DIR/best
echo $BEST_MODEL_DIR
cp /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/model_files/* $BEST_MODEL_DIR
python /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/consensus_classifier/infer_accession_consensus.py --model_dir $BEST_MODEL_DIR --test_csv $DATA_PATH/test.csv --output_csv $OUT_DIR/accession_preds.csv