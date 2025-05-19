import argparse
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score
import ast

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

import importlib.util
import sys
import transformers
"""def import_bert_layers(model_dir):
    model_files_dir = '/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/model_files'
    if model_files_dir not in sys.path:
        sys.path.insert(0, model_files_dir)
    import importlib
    bert_layers = importlib.import_module('bert_layers')
    return bert_layers.BertForSequenceClassification """

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help='Path to finetune/outputs/... directory')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test CSV')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to output CSV (accession, predicted_phenotype, true_phenotype)')
    parser.add_argument('--model_max_length', type=int, default=250)
    parser.add_argument('--run_name', type=str)
    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        args.model_dir,
        num_labels=2,
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_dir,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    # Prepare mapping for phenotype
    pheno_to_idx = {'Resistant': 0, 'Susceptible': 1}
    idx_to_pheno = {v: k for k, v in pheno_to_idx.items()}

    def run_inference(input_csv):
        df = pd.read_csv(input_csv)
        # Use fixed mapping for species for consistency with training
        batch_size = 32
        pred_phenos = []
        with torch.no_grad():
            for start in tqdm(range(0, len(df), batch_size)):
                batch = df.iloc[start:start+batch_size]
                enc = tokenizer(list(batch['sequence']), padding=True, truncation=True, return_tensors='pt', max_length=args.model_max_length)
                num_hits = batch['num_hits'].values
                species = batch['species'].values
                antibiotic = batch['antibiotic'].values

                inputs = {
                    'input_ids': enc['input_ids'].to(device),
                    'attention_mask': enc['attention_mask'].to(device),
                    'num_hits': torch.tensor(num_hits, dtype=torch.float32).unsqueeze(1).to(device),
                    'species': torch.tensor(species, dtype=torch.long).to(device).unsqueeze(1).to(device),
                    'antibiotic': torch.tensor(antibiotic, dtype=torch.long).to(device).unsqueeze(1).to(device),
                }
                logits = model(**inputs).logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                pred_phenos.extend([int(x) for x in preds.flatten()])
        df['pred_phenotype_idx'] = pred_phenos
        return df

    def get_accession_preds(df):
        acc_preds = (
            df.groupby('accession')
            .agg(
                pred_phenotype_idx=('pred_phenotype_idx', lambda x: x.mode().iloc[0]),
                phenotype_idx=('phenotype', 'first'),
                species=('species', 'first')
            )
            .reset_index()
        )
        return acc_preds

    # --- TEST: Apply majority vote ---
    print("Running inference on test set...")
    test_df = run_inference(args.test_csv)

    # Accession-level majority vote
    acc_preds = get_accession_preds(test_df)
    acc_preds['pred_phenotype'] = acc_preds['pred_phenotype_idx'].map(idx_to_pheno)
    acc_preds['true_phenotype'] = acc_preds['phenotype_idx'].map(idx_to_pheno)
    acc_preds[['accession', 'pred_phenotype', 'true_phenotype']].to_csv(args.output_csv, index=False)

    species_mapping = {
    'klebsiella_pneumoniae': 0,
    'streptococcus_pneumoniae': 1,
    'escherichia_coli': 2,
    'campylobacter_jejuni': 3,
    'salmonella_enterica': 4,
    'neisseria_gonorrhoeae': 5,
    'staphylococcus_aureus': 6,
    'pseudomonas_aeruginosa': 7,
    'acinetobacter_baumannii': 8
    }

    species_level_acc = {}
    for species, idx in species_mapping.items():
        species_acc_df = acc_preds[acc_preds['species'] == idx]
        species_level_acc[species] = np.mean(np.array(species_acc_df['pred_phenotype']) == np.array(species_acc_df['true_phenotype']))

    with open(f'{args.run_name}_full_model_species_level_acc.json', 'w') as f:
        json.dump(species_level_acc, f)

    # --- Output test accession stats ---
    test_acc_stats = test_df.groupby('accession').agg(
        num_hits=('num_hits', 'first'),
        num_seqs=('sequence', 'count'),
        num_pred_res=('pred_phenotype_idx', lambda x: (x==0).sum()),
        num_pred_sus=('pred_phenotype_idx', lambda x: (x==1).sum()),
        species=('species', 'first'),
        actual_phenotype=('phenotype', 'first'),
        pred_phenotype=('pred_phenotype_idx', lambda x: x.mode().iloc[0])
    ).reset_index()
    test_acc_stats['pred_phenotype'] = test_acc_stats['pred_phenotype'].map(idx_to_pheno)
    model_name = args.run_name
    test_acc_stats.to_csv(f'/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/consensus_classifier/stats/{model_name}_test_accession_stats.csv', index=False)

    # Evaluate
    acc = (acc_preds['pred_phenotype'] == acc_preds['true_phenotype']).mean()
    print(f'Consensus accession-level accuracy (test set): {acc:.4f}')
    cm = confusion_matrix(acc_preds['true_phenotype'], acc_preds['pred_phenotype'], labels=['Resistant','Susceptible'])
    print("Confusion matrix (rows: true, cols: pred):\n", cm)
    f1 = f1_score(acc_preds['true_phenotype'], acc_preds['pred_phenotype'], labels=['Resistant','Susceptible'], average='macro')
    print(f"Macro F1 score: {f1:.4f}")

    # --- Sequence-level accuracy and confusion ---
    seq_true = test_df['phenotype']
    seq_pred = test_df['pred_phenotype_idx']
    seq_accuracy = (seq_true == seq_pred).mean()
    seq_cm = confusion_matrix(seq_true, seq_pred, labels=[0,1])

    # --- Append summary to CSV ---
    summary_path = "/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/consensus_classifier/2class_performance.csv"
    model_path = args.model_dir
    data_path = args.test_csv
    row = {
        'model_name': run_name,
        'model_path': model_path,
        'data_path': data_path,
        'accession_accuracy': float(acc),
        'accession_confusion': cm.tolist(),
        'accession_f1': float(f1),
        'sequence_accuracy': float(seq_accuracy),
        'sequence_confusion': seq_cm.tolist(),
        
    }
    
    import csv
    import ast
    # If file exists, append, else create
    if os.path.exists(summary_path):
        df_summary = pd.read_csv(summary_path, converters={'accession_confusion': ast.literal_eval, 'sequence_confusion': ast.literal_eval})
        df_summary = pd.concat([df_summary, pd.DataFrame([row])], ignore_index=True)
        df_summary.to_csv(summary_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    else:
        pd.DataFrame([row]).to_csv(summary_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

# --- Visualization code for confusion matrix from summary CSV ---
#
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import ast
#
# summary = pd.read_csv("model_eval_summary.csv", converters={'confusion': ast.literal_eval})
# cm = summary.iloc[-1]['confusion']  # get latest confusion matrix
# labels = ['Resistant', 'Intermediate', 'Susceptible']
# plt.figure(figsize=(6,5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()


if __name__ == '__main__':
    main()
