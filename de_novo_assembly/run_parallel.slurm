#!/bin/bash
#SBATCH -p long-96core        # Partition: long-96core
#SBATCH -J spades   	# Job name
#SBATCH -o res.txt            # Standard output file
#SBATCH -t 48:00:00           # Runtime limit: 48 hours
#SBATCH -c 96                 # Number of CPU cores
#SBATCH --nodes 6
#SBATCH --ntasks-per-node 1

start_time=$(date +%s.%N)

conda activate data_analysis

srun --ntasks=$SLURM_NNODES python -u parallel_denovo_assembly.py --fastq_dir ../data/test_data/fastp --metadata_path ../metadata/test_metadata.csv --isolate --outdir test_assemblies/spades_output_isolate

echo "Script execution time: $duration seconds"
