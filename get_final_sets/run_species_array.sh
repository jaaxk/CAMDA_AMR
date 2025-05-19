#!/bin/bash
#SBATCH -p hbm-long-96core
#SBATCH -o logs/res_%A_%a.txt
#SBATCH -t 48:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=jack.vaska@stonybrook.edu
#SBATCH --array=1-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24

# Array of species to process
SPECIES=(
    "staphylococcus_aureus"
    "salmonella_enterica"
    "acinetobacter_baumannii"
    "campylobacter_jejuni"
)

# Get the species for this array task
CURRENT_SPECIES=${SPECIES[$((SLURM_ARRAY_TASK_ID-1))]}

echo "Processing species: $CURRENT_SPECIES"

export subseq_length=1000
export min_seqs=55
export blast_identity=80
export pvalue_thresh=5e-2

SCRIPT_PATH="/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/dbgwas/run_amr_pipeline.py"
source /gpfs/scratch/jvaska/miniconda3/etc/profile.d/conda.sh

conda activate bioinfo
echo "Running with --dbgwas_sig_level ${pvalue_thresh}"
export FINAL_DATASET_PATH="/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/get_final_sets/final_sets/thresh_${pvalue_thresh}_blast_${blast_identity}_pv4"
export RUN_NAME="thresh_${pvalue_thresh}_blast_${blast_identity}_min_seqs_${min_seqs}_subseq_length_${subseq_length}_pv4"

python -u "$SCRIPT_PATH" --dbgwas_sig_level "$pvalue_thresh" \
--subseq_length "$subseq_length" \
--min_seqs "$min_seqs" \
--blast_identity "$blast_identity" \
--threads 24 \
--species "$CURRENT_SPECIES"

mkdir -p "$FINAL_DATASET_PATH"
echo "Final dataset path: $FINAL_DATASET_PATH"
echo "Run name: $RUN_NAME"
#python /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/get_final_sets/get_final_classifier_set.py
