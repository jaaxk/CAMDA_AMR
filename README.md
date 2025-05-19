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
8. Assemblies were sorted by the 9 species for DBGWAS analysis per species `de_novo_assembly/get_contigs.sh`, `de_novo_assembly/move_files.py`
9. [DBGWAS](https://gitlab.com/leoisl/dbgwas) was run on the assemblies each 9 species, the significant sequences were then extracted and BLASTed back to the original assemblies, and 1000bp subsequences surrounding each sequence were extracted. `dbgwas/run_amr_pipeline.py`
    - For accessions with no BLAST hits (mostly susceptible phenotypes), 55 random 1000bp subsequences were taken from the assembly/reads
    - A dataset was created for each of the 9 species including sequence, num_hits (BLAST hits of significant sequences), accession, species, phenotype
10. Species datasets were pooled together and grouped by antibiotic
    - Invalid seqs were removed (not containing A,C,T or G)
    - Susceptible phenotype rows were augmented 2x by getting reverse complement
    - Data was split into train/test/dev sets with an 80/10/10 split based on randomized accessions in `get_final_sets/train_test_dev_accs` (stratified by phenotype)
    - `export DATA_DIR=path/to/9/species/datasets export FINAL_DATASET_PATH=path/to/output cd get_final_sets/scripts get_final_classifier_set.py per_antibiotic_set.py $FINAL_DATASET_PATH path/to/final/output`
12. A [pretrained DNABERT2 model on human bacterial genomes](https://github.com/jaaxk/DNABERT-M/) (using [MGNify](https://www.ebi.ac.uk/metagenomics) database) was finetuned on the above dataset with extra features `num_hits` and `species`
    - `species` was projected to an 8 dimensional embedding
    - species embedding and num_hits were concatenated with the 768-dim DNABERT2 embedding for finetuning `get_final_sets/run_finetune_ddp_perantibiotic.slurm`
13. Predictions were made by grouping by accession and taking consensus prediction of all rows in accession `consensus_classifier/infer_perantibiotic.py`
 
### Training:
1. Make conda environment from `finetune/dna_env_2.yml`
   - `conda env create -f finetune/dna_env_2.yml`
2. Change BASE_DIR to the base directory of the git repo in `get_final_sets/run_finetune_ddp_perantibiotic.slurm`
3. Pass data directory (containing train.csv, test.csv, and dev.csv) and run name to `get_final_sets/run_finetune_ddp_perantibiotic.slurm`. This script should run training then accession-level evaluation on the best model (based on f1 score on dev set)
   - `cd get_final_sets` 
   - `sbatch run_finetune_ddp_perantibiotic.slurm path/to/data/directory run_name`
   - the data directory should contain 4 subdirectoies called 'TET', 'CAZ', 'GEN', and 'ERY' each with a 'train.csv', 'dev.csv', and 'test.csv' containing the columns 'sequence, num_hits, accession, species, phenotype'

### Inference / Accession-level evaluation
1. To evaluate a model on accession-level accuracy, f1 score, and generate a confusion matrix, run `consensus_classifier/infer_perantibiotic.py` with the test csv, base directory for each 4 models, model max length, output csv, and run name.
2. See `consensus_classifier/infer_perantibiotic.slurm` for an example of how to evaluate a set of models

### Performance
- The per antibiotic models performed with 84.5% accuracy on the testing subset of the training set and 82.8% on the CAMDA-provided test set
- For more, see the [extended abstract]()
