#!/bin/bash

#get accession lists
python metadata/get_accessions.py

#download accessions
./train_data/download_all.slurm

#download failed files
./train_data/download_remaining.slurm

#run fastp
./train_data/run_fastp.slurm

## de novo assembly

#create metadata file
python de_novo_assembly/get_metadata_csv.py

#run assembly
./de_novo_assembly/run_parallel.slurm #will return contigs in de_novo_assembly/contigs
