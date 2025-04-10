import pandas as pd

#these files come with camda download
train_df = pd.read_csv('training_dataset.csv')
test_df = pd.read_csv('testing_dataset_reduced.csv', sep='\t')


train_df['genus_species'] = train_df['genus'] + ' ' + train_df['species']
test_df['genus_species'] = test_df['genus'] + ' ' + test_df['species']

#get this file from train_data/fastq_stats.slurm
fastq_stats_df = pd.read_csv('../test_data/test_fastq_stats.tsv', sep='\t')

fastq_stats_df['file'] = fastq_stats_df['file'].str.split('/').str[-1] #remove path prefix
fastq_stats_df['long_short'] = fastq_stats_df['avg_len'].apply(lambda x: 'long' if x > 500 else 'short') #long or short read
fastq_stats_df['accession'] = fastq_stats_df['file'].str.split('[_.]').str[0]
counts = fastq_stats_df['accession'].value_counts()
fastq_stats_df['layout'] = fastq_stats_df['accession'].apply(lambda x: 'paired' if counts[x] >= 2 else 'single') #single or paired end
fastq_stats_df = fastq_stats_df.merge(
    test_df[['accession', 'genus_species']].drop_duplicates('accession'),
    on='accession', how='left'
) #add genus_species
#fastq_stats_df = fastq_stats_df.merge(
#    train_df[['accession', 'phenotype']].drop_duplicates('accession'),
#    on='accession', how='left'
#) #add phenotype
#fastq_stats_df = fastq_stats_df.merge(
#    train_df[['accession', 'publication']].drop_duplicates('accession'),
#    on='accession', how='left'
#) #add publication

fastq_stats_df['reverse'] = fastq_stats_df.apply(lambda row: fastq_stats_df[(fastq_stats_df['accession'] == row['accession']) & (fastq_stats_df['file'].str.contains('_\d')) & (fastq_stats_df['file'] != row['file'])]['file'].str.extract(r'_(\d+)\.')[0].values[0] if row['layout'] == 'paired' and '_1' in row['file'] else 'NaN', axis=1) #get corresponding reverse read

fastq_stats_df = fastq_stats_df[~((fastq_stats_df['layout'] == 'paired') & (~fastq_stats_df['file'].str.contains('_')))] #get rid of third file for paired end reads

fastq_stats_df.to_csv('test_metadata.csv')

print('Done!')
