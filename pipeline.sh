#!/bin/bash

## Each must be run twice one for train one for test

#get accession lists
python metadata/get_accessions.py

#download accessions
./data/download_all.slurm

#download failed files
./data/download_remaining.slurm

#run fastp
./data/run_fastp.slurm

## de novo assembly

#create metadata file
python de_novo_assembly/get_metadata_csv.py

#run assembly
./de_novo_assembly/run_parallel.slurm #will return contigs in de_novo_assembly/contigs
