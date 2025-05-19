#!/bin/bash

# Get all base filenames (no extension) from files in subdirectories of ../contigs_by_species
found_files=$(find ../train_data/fastp -type f -exec basename {} \; | sed 's/[._].*$//' | sort -u)

echo "Unique accs in dir: " 
find . -type f -exec basename {} \; | sed 's/[._].*$//' | sort -u | wc

# Save to a temporary file
echo "$found_files" > found_files.txt

# Compare with train_accessions.txt and find missing entries
grep -Fxv -f found_files.txt ../train_data/missing_accs.txt > missing_accs_2.txt


