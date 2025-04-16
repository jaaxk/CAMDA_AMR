import pandas as pd

#species_list = ['acinetobacter_baumannii', 'neisseria_gonorrhoeae', 'staphylococcus_aureus', 'streptococcus_pneumoniae', 'escherichia_coli', 'pseudomonas_aeruginosa', 'salmonella_enterica']
species_list = ['klebsiella_pneumoniae']
for species in species_list:
    filename = f'{species}/{species}_train_dataset_finetune.csv'
    df = pd.read_csv(filename)

    df = df.rename(columns={
        'phenotype': 'label'
        })
    mapping = {'Resistant': 0, 'Intermediate': 1, 'Susceptible': 2}
    df['label'] = df['label'].replace(mapping)
    
    print(species)
    print(df.head())
    print(df['label'].value_counts())
    print(len(df))
    df_reduced = df.drop_duplicates(subset=['sequence'])

    print('reduced:')
    print(df_reduced['label'].value_counts())
    print(len(df_reduced))

    df.to_csv(filename, index=False)
    df_reduced.to_csv(filename+'_reduced.csv', index=False)
