import pandas as pd

df = pd.read_csv('train_metadata.csv')
print(df['genus_species'].value_counts())
