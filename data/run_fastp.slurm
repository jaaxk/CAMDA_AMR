#!/bin/bash
#SBATCH -p long-40core        # Partition: long-96core
#SBATCH -J fastp   	# Job name
#SBATCH -o res.txt            # Standard output file
#SBATCH -t 48:00:00           # Runtime limit: 48 hours
#SBATCH --nodes 3
#SBATCH --ntasks-per-node 1

start_time=$(date +%s.%N)

#conda activate data_analysis
srun --ntasks=$SLURM_NNODES python -u parallel_fastp.py --metadata_path ../metadata/train_metadata.csv --accessions train_data/missing_accs.txt --threads 16 --outdir train_data/fastp --indir train_data/fastq

#move any files that didnt fastp:
for f in train_data/fastq/*; do fname=$(basename "$f"); [ ! -e "train_data/fastp/$fname" ] && mv "$f" train_data/fastp/; done

conda activate bioinfo
sbatch ../de_novo_assembly/run_parallel.slurm

duration=$(echo "$end_time - $start_time" | bc -l)
echo "Script execution time: $duration seconds"
