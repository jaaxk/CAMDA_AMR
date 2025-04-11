#!/bin/bash
#SBATCH -p long-96core        # Partition: long-96core
#SBATCH -J stats           # Job name
#SBATCH -o res.txt            # Standard output file
#SBATCH -t 48:00:00           # Runtime limit: 48 hours
#SBATCH -c 96                 # Number of CPU cores
#SBATCH --mem=250G            # Memory: 250GB



seqkit stats ./fastq/*.fastq -T -j 96 > fastq_stats.tsv
