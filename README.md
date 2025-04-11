# Antimicrobial Resistance Prediction using DBGWAS and DNABERT2
## CAMDA AMR Challenge 2025
### Workflow:
1. `training_dataset.csv` and `testing_dataset_reduced.csv` were downloaded from the [CAMDA website](https://bipress.boku.ac.at/camda2025/the-camda-contest-challenges/)
2. `test_accessions.txt` and `train_accessions.txt` were extracted from these datasets using `metadata/get_accessions.py`
3. Reads were downloaded using SRA toolkit's fasterq-dump: `data/download_all.slurm`
4. Metadata file was generated with pandas for information needed for the rest of the workflow: `metadata/get_metadata_csv.py`
5. Quality control and adapter trimming was performed in parallel with fastp across 6 nodes: `data/parallel_fastp.py`
6. Reads were assembled *de novo* with [spades](https://github.com/ablab/spades) with the --isolate flag in parallel across 6 nodes: `de_novo_assembly/parallel_denovo_assembly.py`
7. Assemblies were sorted by the 9 species to create species-specific models downstream `de_novo_assembly/get_contigs.sh`, `de_novo_assembly/move_files.py`
8. [DBGWAS](https://gitlab.com/leoisl/dbgwas) was run on the assemblies each 9 species, the significant sequences were then extracted and BLASTed back to the original assemblies, and 500bp subsequences surrounding each sequence were extracted. Finetuning and classifier datasets were created from these sequences and their associated phenotypes. `dbgwas/run_amr_pipeline.py`
9. The pretrained DNABERT2 model was finetuned 9 times (for each species) ...
