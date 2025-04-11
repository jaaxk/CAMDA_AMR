#!/bin/bash
#SBATCH -p long-40core        # Partition: long-96core
#SBATCH -J fastp   	# Job name
#SBATCH -o res.txt            # Standard output file
#SBATCH -t 48:00:00           # Runtime limit: 48 hours
#SBATCH --nodes 3
#SBATCH --ntasks-per-node 1

start_time=$(date +%s.%N)

conda activate data_analysis
srun --ntasks=$SLURM_NNODES python -u parallel_fastp.py --metadata_path ../metadata/test_metadata.csv --accessions test_data/test_accessions.txt --threads 16 --outdir test_data/fastp --indir test_data/fastq

#move any files that didnt fastp:
for f in test_data/fastq/*; do fname=$(basename "$f"); [ ! -e "test_data/fastp/$fname" ] && mv "$f" test_data/fastp/; done

conda activate bioinfo
sbatch ../de_novo_assembly/run_parallel.slurm

duration=$(echo "$end_time - $start_time" | bc -l)
echo "Script execution time: $duration seconds"
