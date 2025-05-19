#!/bin/bash
#SBATCH -p hbm-long-96core
#SBATCH --time 48:00:00
#SBATCH --output res.txt

export CONTIGS_DIR="/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/de_novo_assembly/test_assemblies/contigs_by_species"
export BASE_DIR="/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR"
export METADATA="/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/metadata/testing_template.csv"
export SUBSEQ_LENGTH=1000
export MIN_SEQS=55
export BLAST_IDENTITY=80
export DBGWAS_SIG_LEVEL=5e-2
export RUN_NAME=TEST_CLASSIFIER_NO_TOPUP

#find . -type d -name "dbs" -exec rm -r {} +
#find . -type d -name "blast_output" -exec rm -r {} +
#find . -type f -name "*sig_sequences.fasta" -exec rm {} +
#find . -name "*_regression.csv" -exec rm {} +


#python -u run_amr_pipeline.py --threads 96 --metadata $METADATA --subseq_length $SUBSEQ_LENGTH --min_seqs $MIN_SEQS --blast_identity $BLAST_IDENTITY --dbgwas_sig_level $DBGWAS_SIG_LEVEL --contigs_dir $CONTIGS_DIR --test_set

find $BASE_DIR/dbgwas -type f -name "*_classifier.csv" -exec mv {} $BASE_DIR/get_final_sets/datasets_by_species/$RUN_NAME \;

export FINAL_DATASET_PATH=$BASE_DIR/get_final_sets/final_sets/$RUN_NAME
mkdir -p $FINAL_DATASET_PATH
export DATA_DIR=$BASE_DIR/get_final_sets/datasets_by_species/$RUN_NAME
python $BASE_DIR/get_final_sets/scripts/get_final_classifier_test_set.py 

python $BASE_DIR/get_final_sets/scripts/per_antibiotic_set_test.py $DATA_DIR ${DATA_DIR}_PER_ANTIBIOTIC

sbatch run_finetune_ddp_perantibiotic.slurm 
