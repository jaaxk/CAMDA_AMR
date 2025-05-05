#/bin/bash
#SBATCH -p long-96core        # Partition: long-96core
#SBATCH -J download           # Job name
#SBATCH -o res.txt            # Standard output file
#SBATCH -t 48:00:00           # Runtime limit: 48 hours
#SBATCH -c 96                 # Number of CPU cores

cat test_data/missing_accs_test.txt | xargs -n 1 -P 96 fasterq-dump --split-files --skip-technical --outdir ./test_data/fastq
