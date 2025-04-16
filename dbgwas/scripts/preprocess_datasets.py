import pandas as pd
import glob

csv_files = glob.glob("datasets/*.csv")

for file in csv_files:
    df = pd.read_csv(file)

    final_df = df.drop_duplicates(subset=['sequence'], keep=False)
    counts = df['sequence'].value_counts().reset_index()
    new_rows = []
    for seq in counts[counts['count'] > 1]['sequence']:
        proportions = df['label'].value_counts(normalize=True).to_dict()
        proportions = {k: 1-v for k,v in proportions.items()}
        seq_counts = df[df['sequence'] == seq]['label'].value_counts().to_dict()
        norm_seq_counts = {k: v*proportions[k] for k,v in seq_counts.items()}
        label = max(norm_seq_counts, key=norm_seq_counts.get)
        new_rows.append([seq, label])

    final_df = pd.concat([pd.DataFrame(new_rows, columns=['sequence', 'label']), final_df], ignore_index=True)
    final_df.to_csv(file+'_reduced.csv', index=False)
