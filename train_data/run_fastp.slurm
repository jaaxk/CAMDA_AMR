#!/bin/bash
#SBATCH -p long-40core        # Partition: long-96core
#SBATCH -J fastp   	# Job name
#SBATCH -o res.txt            # Standard output file
#SBATCH -t 48:00:00           # Runtime limit: 48 hours
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 1

start_time=$(date +%s.%N)

conda activate data_analysis
srun --ntasks=$SLURM_NNODES python -u parallel_fastp.py --metadata_path ../metadata/train_metadata.csv --accessions train_accessions.txt --threads 16 --outdir fastp

duration=$(echo "$end_time - $start_time" | bc -l)
echo "Script execution time: $duration seconds"
