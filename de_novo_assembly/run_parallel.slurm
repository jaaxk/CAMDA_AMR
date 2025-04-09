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

srun --ntasks=$SLURM_NNODES python -u parallel_denovo_assembly.py --fastq_dir ../train_data/fastp --metadata_path ../metadata/train_metadata.csv --isolate --outdir spades_output_isolate

#extract contig files:
CONTIGS_DIR="contigs"
SPADES_DIR="spades_output"
mkdir -p "$CONTIGS_DIR"

for subdir in "$SPADES_DIR"/*; do
    if [ -d "$subdir" ]; then  # Check if it's a directory
        # Extract the subdirectory name (without path)
        dirname=$(basename "$subdir")

        # Define the contigs file path
        contigs_file="$subdir/contigs.fa"

        # Check if 'contigs.fa' exists
        if [ -f "$contigs_file" ]; then
            new_filename="$CONTIGS_DIR/${dirname}.fa"
            mv "$contigs_file" "$new_filename" 
            echo "Copied $contigs_file to $new_filename"
        else
            echo "Warning: 'contigs.fa' not found in $subdir"
        fi
    fi
done
echo "Done! All contigs moved to '$CONTIGS_DIR/'."

duration=$(echo "$end_time - $start_time" | bc -l)
echo "Script execution time: $duration seconds"
