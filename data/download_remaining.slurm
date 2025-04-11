#!/bin/bash
#SBATCH -p long-96core        # Partition: long-96core
#SBATCH -J download           # Job name
#SBATCH -o res.txt            # Standard output file
#SBATCH -t 48:00:00           # Runtime limit: 48 hours
#SBATCH -c 96                 # Number of CPU cores

FASTQ_DIR="./fastq"

while read accession; do
	if ls $FASTQ_DIR/${accession}* 1> /dev/null 2>&1; then
   		 echo "Files for $accession already exist. Skipping."
 	 else
   		 echo "Downloading $accession..."
	  	 fasterq-dump --split-files --skip-technical --outdir $FASTQ_DIR $accession

		if [ $? -eq 0 ]; then
      			echo "Successfully downloaded $accession"
    		else
      			echo "Failed to download $accession" >&2
    		fi
  	fi
done < train_accessions.txt
