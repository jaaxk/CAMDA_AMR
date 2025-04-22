# Antimicrobial Resistance Prediction using DBGWAS and DNABERT2
## CAMDA AMR Challenge 2025
### Workflow:
1. `training_dataset.csv` and `testing_dataset_reduced.csv` were downloaded from the [CAMDA website](https://bipress.boku.ac.at/camda2025/the-camda-contest-challenges/)
2. `test_accessions.txt` and `train_accessions.txt` were extracted from these datasets using `metadata/get_accessions.py`
3. Reads were downloaded using SRA toolkit's fasterq-dump: `data/download_all.slurm`
4. Metadata file was generated with pandas for information needed for the rest of the workflow: `metadata/get_metadata_csv.py`
5. Quality control and adapter trimming was performed in parallel with fastp across 6 nodes: `data/parallel_fastp.py`
6. Reads were assembled *de novo* with [spades](https://github.com/ablab/spades) with the --isolate flag in parallel across 6 nodes: `de_novo_assembly/parallel_denovo_assembly.py`
  - For accessions that de novo assembly failed, one of the raw reads files was used in place of an assembly
8. Assemblies were sorted by the 9 species to create species-specific models downstream `de_novo_assembly/get_contigs.sh`, `de_novo_assembly/move_files.py`
9. [DBGWAS](https://gitlab.com/leoisl/dbgwas) was run on the assemblies each 9 species, the significant sequences were then extracted and BLASTed back to the original assemblies, and 500bp subsequences surrounding each sequence were extracted. `dbgwas/run_amr_pipeline.py`
  - For accessions with no BLAST hits (mostly susceptible phenotypes), 9 random 500bp subsequences were taken from the assembly/reads
  - A dataset was created for each of the 9 species including sequence, num_hits (BLAST hits of significant sequences), accession, species, phenotype
10. Duplicates (subset accession and sequence) were removed and all 9 datasets were combined into one. `dbgwas/scripts/get_full_classifier_dataset.py`
    - Invalid seqs were removed (not containing A,C,T or G)
    - Rows with resistant phenotype were sampled to 50% while stratifying for accession and ensuring no accession was rid of entirely
    - Susceptible phenotype rows were augmented 2x by getting reverse complement
    - Intermediate phenotype rows were augmented 4x by getting reverse complement and complement
    - Train/test split of 85/15 was done by stratifying on phenotype and ensuring accessions were not shared across sets (for consensus classifier)
11. A [pretrained DNABERT2 model on human bacterial genomes](https://github.com/jaaxk/DNABERT-M/) (using [MGNify](https://www.ebi.ac.uk/metagenomics) database) was finetuned on the above dataset with extra features `num_hits` and `species`
    - Class weights were given to the loss function
    - `species` was turned into a 4-dimensional embedding
    - species embedding and num_hits were concatenated with the 768-dim DNABERT2 embedding
12. A 'consensus classifier' was created to predict a phenotype for each row in an accession and get the maximum of the predicted phenotypes per accession
