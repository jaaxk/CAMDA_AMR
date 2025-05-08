#!/bin/bash
#SBATCH -p hbm-long-96core
#SBATCH -o logs/res_%j.txt
#SBATCH -t 48:00:00
#SBATCH --ntasks-per-node=4
#SBATCH -c 24
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=jack.vaska@stonybrook.edu

export subseq_length=1000
export min_seqs=55
export blast_identity=80
export pvalue_thresh=5e-2

SCRIPT_PATH="/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/dbgwas/run_amr_pipeline.py"
source /gpfs/scratch/jvaska/miniconda3/etc/profile.d/conda.sh #for conda envs to work
conda activate bioinfo

echo "Running with --dbgwas_sig_level ${pvalue_thresh}"
export FINAL_DATASET_PATH="/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/get_final_sets/final_sets/thresh_${pvalue_thresh}_blast_${blast_identity}_pv4"
export RUN_NAME="thresh_${pvalue_thresh}_blast_${blast_identity}_min_seqs_${min_seqs}_subseq_length_${subseq_length}_pv4"

for species in escherichia_coli pseudomonas_aeruginosa; do
    logfile="logs/${species}_%j.log"
    echo "Launching job for species: $species"
    python -u "$SCRIPT_PATH" \
        --dbgwas_sig_level "$pvalue_thresh" \
        --subseq_length "$subseq_length" \
        --min_seqs "$min_seqs" \
        --blast_identity "$blast_identity" \
        --threads 24 \
        --species "$species" &
done
wait
