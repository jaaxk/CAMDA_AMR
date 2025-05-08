#!/bin/bash
#SBATCH -p hbm-long-96core
#SBATCH -o logs/res_%j.txt
#SBATCH -t 12:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=jack.vaska@stonybrook.edu
#SBATCH --mem=350G

export min_seqs=55
export blast_identity=80
export sig_threshold=5e-2

SUBSEQ_LENGTHS=(2000 4000 500)

SCRIPT_PATH="/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/dbgwas/run_amr_pipeline.py"
source /gpfs/scratch/jvaska/miniconda3/etc/profile.d/conda.sh #for conda envs to work
find /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/dbgwas -name "blast_output" -exec rm -rf {} +
find /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/dbgwas -name "*sig_sequences.fasta" -exec rm -f {} +
find /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/dbgwas -name "_classifier.csv" -exec rm -f {} +

for subseq_length in "${SUBSEQ_LENGTHS[@]}"; do
    conda activate bioinfo
    echo "Running with --subseq_length ${subseq_length}"
    export FINAL_DATASET_PATH="/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/get_final_sets/final_sets/subseq_length_thresh_${sig_threshold}_${subseq_length}_pv4_mv4"
    export RUN_NAME="subseq_length_thresh_${sig_threshold}_${subseq_length}_pv4_mv4"
    #find /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/dbgwas -name "*sig_sequences.fasta" -exec rm -f {} +

    python -u "$SCRIPT_PATH" --dbgwas_sig_level "$sig_threshold" \
    --subseq_length "$subseq_length" \
    --min_seqs "$min_seqs" \
    --blast_identity "$blast_identity"

    if [ $? -ne 0 ]; then
        echo "Python script failed, quitting"
        exit 1
    fi

    export DATA_DIR="/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/get_final_sets/datasets_by_species/$RUN_NAME"
    mkdir -p "$DATA_DIR"
    find /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/dbgwas -name "*train_dataset_classifier.csv" -exec mv -t "$DATA_DIR" {} +
    mkdir -p "$FINAL_DATASET_PATH"
    echo "Final dataset path: $FINAL_DATASET_PATH"
    python /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/get_final_sets/get_final_classifier_set.py

    #Run finetuning
    conda activate dna
    sbatch /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/get_final_sets/run_finetune_ddp.slurm "$FINAL_DATASET_PATH" "$RUN_NAME"



done
