#!/bin/bash

# Set the root directory to search
SEARCH_DIR="$1"

# Set the destination directory for FASTQ or unknown files
FASTQ_DIR="$SEARCH_DIR/fastq"

# Create the destination directory if it doesn't exist
mkdir -p "$FASTQ_DIR"

# Function to detect file format
detect_format() {
    local file="$1"
    
    # Read the first non-empty line
    first_line=$(grep -m 1 -v '^$' "$file")
    
    # Check the first character
    if [[ "$first_line" == @* ]]; then
        echo "FASTQ"
    elif [[ "$first_line" == '>'* ]]; then
        echo "FASTA"
    else
        echo "UNKNOWN"
    fi
}

# Export function so it's usable in subshells
export -f detect_format

# Export variables
export FASTQ_DIR

# Find all regular files in the directory
find "$SEARCH_DIR" -type f | while read -r file; do
    format=$(detect_format "$file")
    if [[ "$format" == "FASTQ" || "$format" == "UNKNOWN" ]]; then
        echo "Moving $format file: $file"
        mv "$file" "$FASTQ_DIR/"
    fi
done

