#!/bin/bash

# Loop through all .fasta files in current directory and subdirectories
find . -type f -name "*.fasta" | while read -r file; do
    echo "Checking file: $file"

    # Check if file looks like it's in FASTQ format
    line1=$(head -n 1 "$file")
    line3=$(head -n 3 "$file" | tail -n 1)

    if [[ "$line1" == @* && "$line3" == "+"* ]]; then
        echo "Detected FASTQ format. Converting to FASTA and replacing original..."

        # Use a temporary file to store the converted output
        tmp_file=$(mktemp)

        # Convert using seqkit
        if seqkit fq2fa "$file" -o "$tmp_file"; then
            mv "$tmp_file" "$file"
            echo "Successfully replaced: $file"
        else
            echo "Conversion failed for: $file"
            rm -f "$tmp_file"
        fi
    else
        echo "File is not in FASTQ format. Skipping..."
    fi
done

