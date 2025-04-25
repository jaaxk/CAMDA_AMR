#!/bin/bash
#SBATCH -p long-40core        # Partition: long-96core
#SBATCH -J spades   	# Job name
#SBATCH -o res.txt            # Standard output file
#SBATCH -t 48:00:00           # Runtime limit: 48 hours
#SBATCH -c 40                 # Number of CPU cores
#SBATCH --nodes 6
#SBATCH --ntasks-per-node 1

start_time=$(date +%s.%N)

conda activate bioinfo

srun --ntasks=$SLURM_NNODES python -u parallel_denovo_assembly.py --fastq_dir /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/data/train_data/fastp --metadata_path /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/metadata/train_metadata.csv --isolate --outdir /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/de_novo_assembly/spades_output_isolate

echo "Removing: "

find /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/de_novo_assembly/spades_output_isolate -mindepth 1 -maxdepth 1 -type d ! -exec test -f "{}/contigs.fasta" \; -print | wc

find /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/de_novo_assembly/spades_output_isolate -mindepth 1 -maxdepth 1 -type d ! -exec test -f "{}/contigs.fasta" \; -exec rm -r {} +

srun --ntasks=$SLURM_NNODES python -u parallel_denovo_assembly.py --fastq_dir /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/data/train_data/fastp --metadata_path /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/metadata/train_metadata.csv --isolate --outdir /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/de_novo_assembly/spades_output_isolate --single_end

./gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/de_novo_assembly/scripts/get_contigs.sh
#python /gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/de_novo_assembly/scripts/move_files.py #Quality check contigs folder first before moving to species dirs

echo "Script execution time: $duration seconds"
