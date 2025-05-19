#!/bin/bash
#SBATCH -p hbm-long-96core
#SBATCH -o logs/res_%j.txt
#SBATCH -t 48:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=jack.vaska@stonybrook.edu
#SBATCH --mem 350G

export subseq_length=1000
export min_seqs=1
export blast_identity=80
export pvalue_thresh=5e-2

SCRIPT_PATH="/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/dbgwas/run_amr_pipeline.py"
source /gpfs/scratch/jvaska/miniconda3/etc/profile.d/conda.sh #for conda envs to work

conda activate bioinfo
echo "Running with --dbgwas_sig_level ${pvalue_thresh}"
export FINAL_DATASET_PATH="/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/get_final_sets/final_sets/thresh_${pvalue_thresh}_blast_${blast_identity}_minseqs_${min_seqs}pv4"
export RUN_NAME="thresh_${pvalue_thresh}_blast_${blast_identity}_min_seqs_${min_seqs}_subseq_length_${subseq_length}_pv4"
#rm -f /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/get_final_sets/datasets_by_species/*
find /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/dbgwas -name "*sig_sequences.fasta" -exec rm -f {} +
find /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/dbgwas -name "blast_output" -exec rm -rf {} +
find /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/dbgwas -name "_classifier.csv" -exec rm -f {} +

python -u "$SCRIPT_PATH" --dbgwas_sig_level "$pvalue_thresh" \
--subseq_length "$subseq_length" \
--min_seqs "$min_seqs" \
--blast_identity "$blast_identity" \
--threads 96 \

if [ $? -ne 0 ]; then
	echo "Python script failed, quitting"
        exit 1
fi
#rm -f /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/get_final_sets/datasets_by_species/*
#find /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/dbgwas -name "*train_dataset_classifier.csv" -exec mv -t /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/get_final_sets/datasets_by_species/ {} +

export DATA_DIR="/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/get_final_sets/datasets_by_species/$RUN_NAME"
mkdir -p "$DATA_DIR"
find /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/dbgwas -name "*train_dataset_classifier.csv" -exec mv -t "$DATA_DIR" {} +
mkdir -p "$FINAL_DATASET_PATH"
echo "Final dataset path: $FINAL_DATASET_PATH"
python /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/get_final_sets/get_final_classifier_set.py

#Run finetuning
conda activate dna
sbatch /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/get_final_sets/run_finetune_cpu.slurm "$FINAL_DATASET_PATH" "$RUN_NAME"

