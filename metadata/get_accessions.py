import pandas as pd

#these files come with camda download
train_df = pd.read_csv('training_dataset.csv')
test_df = pd.read_csv('testing_dataset_reduced.csv', sep='\t')i

#get train_accessions.txt for download
with open("train_accessions.txt", 'w') as f:
    for accession in train_df['accession']:
        f.write(accession+'\n')

with open("test_accessions.txt", 'w') as f:
    for accession in test_df['accession']:
        f.write(accession+'\n')
