#!/bin/bash

# Output file to store names of .fasta files with max_len < 1000
output_file="/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/de_novo_assembly/quality_check/short_fasta_files.txt"
> "$output_file"  # Clear the file if it exists

# Iterate through all .fasta files in the current directory
for fasta in *.fasta; do
    # Check if file exists to handle the case when no .fasta files are found
    [ -e "$fasta" ] || continue

    # Get the max length using seqkit stats (skip header, extract 5th column)
    max_len=$(seqkit stats "$fasta" | awk 'NR==2 {gsub(",", "", $5); print $5}')
    # Check if max_len is less than 1000
    if [ "$max_len" -lt 2000 ]; then
        # Strip extension and write basename to output file
        basename="${fasta%.fasta}"
        echo "$basename" >> "$output_file"
    fi
done

echo "Done. Results saved in $output_file"
