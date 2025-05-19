#!/bin/bash

# Output file
output_file="invalid_fasta_files.txt"
> "$output_file"  # clear previous contents

# Loop through .fastq files one subdirectory deep
for fasta in ../train_data/fastp/*.fastq; do
    # Run seqkit stats and capture output (both stdout and stderr)
    stats=$(seqkit stats "$fasta" 2>&1)

    base_name=$(basename "$fasta" .fastq)

    # Check if seqkit failed or gave a bad format error
    if [[ $? -ne 0 || "$stats" == *"fastx: bad fastq format"* ]]; then
        echo "Invalid or unreadable format: $fasta"
        echo "$base_name" >> "$output_file"
        continue
    fi

    # Extract number of sequences (skip header line)
    num_seqs=$(echo "$stats" | awk 'NR==2 {print $4}')

    # If the number of sequences is 0, it's invalid
    if [[ "$num_seqs" -eq 0 ]]; then
        echo "No sequences in $fasta"
        echo "$base_name" >> "$output_file"
    fi
done

