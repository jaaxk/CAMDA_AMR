#!/bin/bash
#SBATCH -p
#SBATCH -o res.txt
#SBATCH -t 12:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=jack.vaska@stonybrook.edu

export subseq_length=1000
export min_seqs=55
export blast_identity=80

#SIG_THRESHOLDS=(5e-2 5e-4 5e-6 5e-8 5e-10 5e-12)

SIG_THRESHOLDS=(1.0 5e-1 5e-3 5e-10 5e-12)

SCRIPT_PATH="/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/dbgwas/run_amr_pipeline.py"
source /gpfs/scratch/jvaska/miniconda3/etc/profile.d/conda.sh #for conda envs to work

for THRESH in "${SIG_THRESHOLDS[@]}"; do
    conda activate bioinfo
    echo "Running with --dbgwas_sig_level ${THRESH}"
    export FINAL_DATASET_PATH="/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/tune_hyperparams/final_sets/thresh_${THRESH}_blast_${blast_identity}_pv2"
    export RUN_NAME="thresh_${THRESH}_blast_${blast_identity}_min_seqs_${min_seqs}_subseq_length_${subseq_length}_pv2"
    rm -f /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/tune_hyperparams/datasets_by_species/*
    find /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/dbgwas -name "*sig_sequences.fasta" -exec rm -f {} +
    find /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/dbgwas -name "blast_output" -exec rm -rf {} +

    python -u "$SCRIPT_PATH" --dbgwas_sig_level "$THRESH" \
    --subseq_length "$subseq_length" \
    --min_seqs "$min_seqs" \
    --blast_identity "$blast_identity"

    rm -f /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/tune_hyperparams/datasets_by_species/*
    find /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/dbgwas -name "*train_dataset_classifier.csv" -exec mv -t /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/tune_hyperparams/datasets_by_species/ {} +
    mkdir -p "$FINAL_DATASET_PATH"
    echo "Final dataset path: $FINAL_DATASET_PATH"
    python /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/tune_hyperparams/get_final_classifier_set.py

    #Run finetuning
    conda activate dna
    sbatch /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/tune_hyperparams/run_finetune_ddp.slurm "$FINAL_DATASET_PATH" "$RUN_NAME"



done
