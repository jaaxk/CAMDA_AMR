#!/bin/bash

#extract contig files: CHECK THESE FOR TRAIN/TEST
CONTIGS_DIR="/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/de_novo_assembly/contigs"
SPADES_DIR="/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/de_novo_assembly/spades_output_isolate"
mkdir -p "$CONTIGS_DIR"

for subdir in "$SPADES_DIR"/*; do
    if [ -d "$subdir" ]; then  # Check if it's a directory
        # Extract the subdirectory name (without path)
        dirname=$(basename "$subdir")

        # Define the contigs file path
        contigs_file="$subdir/contigs.fasta"

        # Check if 'contigs.fa' exists
        if [ -f "$contigs_file" ]; then
            new_filename="$CONTIGS_DIR/${dirname}.fasta"
            mv "$contigs_file" "$new_filename"
            echo "Copied $contigs_file to $new_filename"
        else
            echo "Warning: 'contigs.fa' not found in $subdir"

        fi
    fi
done
echo "Done! All contigs moved to '$CONTIGS_DIR/'."
