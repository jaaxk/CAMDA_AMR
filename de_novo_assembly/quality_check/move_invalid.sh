#!/bin/bash

# Make output directory if it doesn't exist
mkdir -p invalid_assemblies

# Loop through each accession in the list
while read -r acc; do
    # Search for any matching file in the subdirectories
    match=$(find ../contigs_by_species -type f | grep -E "/[^/]*$acc(\.[^.]+)?$")

    if [[ -n "$match" ]]; then
        echo "Moving match for $acc:"
        echo "$match"
        # Move each match (in case there's more than one)
        while read -r file; do
            mv "$file" invalid_assemblies/
        done <<< "$match"
    else
        echo "❗ No match found for accession: $acc"
    fi
done < invalid_fasta_files.txt

