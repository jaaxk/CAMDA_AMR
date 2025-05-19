#!/bin/bash

# Output file
output_file="invalid_fasta_files_test.txt"
> "$output_file"  # clear previous contents

# Loop through .fasta files one subdirectory deep
for fasta in ../test_assemblies/contigs/*.fasta; do
    # Run seqkit stats and capture output
    stats=$(seqkit stats "$fasta" 2>/dev/null)
    
    # Check if seqkit failed to read the file
    if [[ $? -ne 0 || "$stats" == *"fastx: bad fastq format"* ]]; then
        echo "seqkit failed to read $fasta"
        base_name=$(basename "$fasta" .fasta)
        echo "$base_name" >> "$output_file"
        continue
    fi

    # Extract number of sequences (skip header line)
    num_seqs=$(echo "$stats" | awk 'NR==2 {print $4}')

    # If the number of sequences is 0, it's invalid
    if [[ "$num_seqs" -eq 0 ]]; then
        echo "No sequences in $fasta"
        base_name=$(basename "$fasta" .fasta)
        echo "$base_name" >> "$output_file"
    fi
done

