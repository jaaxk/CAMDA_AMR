#!/bin/bash

# Get all base filenames (no extension) from files in subdirectories of ../contigs_by_species
found_files=$(find ../test_assemblies/contigs -type f -exec basename {} \; | sed 's/\.[^.]*$//' | sort -u)

# Save to a temporary file
echo "$found_files" > found_files.txt

# Compare with train_accessions.txt and find missing entries
grep -Fxv -f found_files.txt ../../data/test_data/test_accessions.txt > missing_accs_test.txt


