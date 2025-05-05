#!/bin/bash
#SBATCH -p long-96core        # Partition: long-96core
#SBATCH -J download           # Job name
#SBATCH -o res.txt            # Standard output file
#SBATCH -t 48:00:00           # Runtime limit: 48 hours
#SBATCH -c 96                 # Number of CPU cores


FASTQ_DIR="./train_data/fastq"
MISSING_ACC_FILE="./train_data/missing_accs.txt"
TOTAL_ACC=$(wc -l < "$MISSING_ACC_FILE")

# Loop until number of unique downloaded files equals the number of accessions to download
while [ "$(find "$FASTQ_DIR" -type f -exec basename {} \; | sed 's/[._].*$//' | sort -u | wc -l)" -lt "$TOTAL_ACC" ]; do
  while read accession; do
    # Check if any file for this accession already exists
    if ls "$FASTQ_DIR/${accession}"* 1> /dev/null 2>&1; then
      echo "Files for $accession already exist. Skipping."
    else
      echo "Downloading $accession..."
      fasterq-dump --split-files --skip-technical --outdir "$FASTQ_DIR" "$accession"

      if [ $? -eq 0 ]; then
        echo "Successfully downloaded $accession"
      else
        echo "Failed to download $accession" >&2
      fi
    fi
  done < "$MISSING_ACC_FILE"
done

