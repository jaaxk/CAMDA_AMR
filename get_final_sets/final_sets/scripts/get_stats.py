import pandas as pd
import sys

set_name = sys.argv[1]

train_df = pd.read_csv(f'/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/get_final_sets/final_sets/{set_name}/train.csv')
test_df = pd.read_csv(f'/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/get_final_sets/final_sets/{set_name}/test.csv')

print('Train length: ', len(train_df))
print('Test length: ', len(test_df))

print('Train pheno counts: ')
print(train_df['phenotype'].value_counts())

print('Test pheno counts: ')
print(test_df['phenotype'].value_counts())

print('Num hits distribution: ')
print(train_df['num_hits'].describe())

print('Num seqs with 0 hits: ')
print(len(train_df[train_df['num_hits']==0]))
