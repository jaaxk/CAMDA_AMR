#!/bin/bash

./get_invalid_acc.sh

#Get train accessions from original CAMDA dataset like this:
#unique_accessions = og_train['accession'].unique()

#with open("train_accessions.txt", "w") as f:
#    for accession in unique_accessions:
#        f.write(f"{accession}\n")


#CHECK invalid_fasta_files.txt before moving on

./move_invalid.sh

./get_missing_accs.sh

cp missing_accs.txt ../../data/train_data/

./ ../../data/download_all.slurm
