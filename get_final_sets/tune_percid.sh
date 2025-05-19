#!/bin/bash
#SBATCH -p long-96core
#SBATCH -o res_percid.txt
#SBATCH -t 48:00:00

export subseq_length=1000
export min_seqs=55
export sig_threshold=5e-12

BLAST_IDENTITIES=(40 60 70 85 95)
SCRIPT_PATH="/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/dbgwas/run_amr_pipeline.py"
source /gpfs/scratch/jvaska/miniconda3/etc/profile.d/conda.sh #for conda envs to work

for blast_identity in "${BLAST_IDENTITIES[@]}"; do
    conda activate bioinfo
    echo "Running with --blast_identity ${blast_identity} and --dbgwas_sig_level ${sig_threshold}"
    export RUN_NAME="thresh_${sig_threshold}_blast_${blast_identity}_min_seqs_${min_seqs}_subseq_length_${subseq_length}"
    export FINAL_DATASET_PATH="/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/tune_hyperparams/final_sets/${RUN_NAME}"
    rm -f /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/tune_hyperparams/datasets_by_species/*
    find /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/dbgwas/species -name "*sig_sequences.fasta" -exec rm -f {} +
    find /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/dbgwas/species -name "blast_output" -exec rm -rf {} +

    python -u "$SCRIPT_PATH" --dbgwas_sig_level "$sig_threshold" \
    --subseq_length "$subseq_length" \
    --min_seqs "$min_seqs" \
    --blast_identity "$blast_identity"

    rm -f /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/get_final_sets/datasets_by_species/*
    find /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/dbgwas/species -name "*train_dataset_classifier.csv" -exec mv -t /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/get_final_sets/datasets_by_species/ {} +
    mkdir -p "$FINAL_DATASET_PATH"
    echo "Final dataset path: $FINAL_DATASET_PATH"
    python /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/get_final_sets/get_final_classifier_set.py

    # Run finetuning
    conda activate dna
    sbatch /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/tune_hyperparams/run_finetune_ddp.slurm "$FINAL_DATASET_PATH" "$RUN_NAME"

done
