import argparse
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score
import ast
import csv
import json


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import importlib.util
import sys
import transformers
import gc

def run_inference(input_df, device, model, tokenizer, args, species):
    batch_size = 32
    pred_phenos = []
    with torch.no_grad():
        for start in tqdm(range(0, len(input_df), batch_size)):
            batch = input_df.iloc[start:start+batch_size]
            enc = tokenizer(list(batch['sequence']), padding=True, truncation=True, return_tensors='pt', max_length=args.model_max_length)
            num_hits = batch['num_hits'].values
            antibiotic = batch['antibiotic'].values


            if species == 'pseudomonas_aeruginosa' or species == 'neisseria_gonorrhoeae':
                species_idx = batch['species'].values
                inputs = {
                'input_ids': enc['input_ids'].to(device),
                'attention_mask': enc['attention_mask'].to(device),
                'num_hits': torch.tensor(num_hits, dtype=torch.float32).unsqueeze(1).to(device),
                'antibiotic': torch.tensor(antibiotic, dtype=torch.long).to(device).unsqueeze(1).to(device),
                'species': torch.tensor(species_idx, dtype=torch.long).to(device).unsqueeze(1).to(device),
            }
            else:
                inputs = {
                'input_ids': enc['input_ids'].to(device),
                'attention_mask': enc['attention_mask'].to(device),
                'num_hits': torch.tensor(num_hits, dtype=torch.float32).unsqueeze(1).to(device),
                'antibiotic': torch.tensor(antibiotic, dtype=torch.long).to(device).unsqueeze(1).to(device),
                }
            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            pred_phenos.extend([int(x) for x in preds.flatten()])

    input_df['pred_phenotype_idx'] = pred_phenos
    return input_df


def get_accession_preds(df, idx_to_pheno):
    acc_preds = (
        df.groupby('accession')
        .agg(
            pred_phenotype_idx=('pred_phenotype_idx', lambda x: x.mode().iloc[0]),
            phenotype_idx=('phenotype', 'first')
        )
        .reset_index()
    )
    acc_preds['pred_phenotype'] = acc_preds['pred_phenotype_idx'].map(idx_to_pheno)
    acc_preds['true_phenotype'] = acc_preds['phenotype_idx'].map(idx_to_pheno)
    return acc_preds


def load_model(model_path, args, device):
    
    torch.cuda.empty_cache()
    gc.collect()

    config = transformers.AutoConfig.from_pretrained(
        model_path,
        num_labels=2,
        trust_remote_code=True,
    )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    return model, tokenizer

def get_stats(test_df, args, idx_to_pheno, species):
    test_acc_stats = test_df.groupby('accession').agg(
        num_hits=('num_hits', 'first'),
        num_seqs=('sequence', 'count'),
        num_pred_res=('pred_phenotype_idx', lambda x: (x==0).sum()),
        num_pred_sus=('pred_phenotype_idx', lambda x: (x==1).sum()),
        actual_phenotype=('phenotype', 'first'),
        pred_phenotype=('pred_phenotype_idx', lambda x: x.mode().iloc[0])
    ).reset_index()
    test_acc_stats['pred_phenotype'] = test_acc_stats['pred_phenotype'].map(idx_to_pheno)
    test_acc_stats['species'] = species
    model_name = args.model_dir.split('/')[-2]
    test_acc_stats.to_csv(f'{args.output_csv}_{model_name}_stats.csv', index=False)







def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', type=str, required=True, help='Path to base directory containing directories for each model')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to SINGLE test csv within base directory (containing all species)')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to output CSV (accession, predicted_phenotype, true_phenotype)')
    parser.add_argument('--model_max_length', type=int, default=250)
    parser.add_argument('--final_set', action='store_true', default=False)
    parser.add_argument('--summary_path', type=str, default='/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/consensus_classifier/2class_performance.csv')
    parser.add_argument('--run_name', type=str, required=False, default='norunname_')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pheno_to_idx = {'Resistant': 0, 'Susceptible': 1}
    idx_to_pheno = {v: k for k, v in pheno_to_idx.items()}

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

    #Load CSV:
    df = pd.read_csv(args.test_csv)
    
    total_acc_true = []
    total_acc_pred = []
    total_seq_true = []
    total_seq_pred = []
    per_species_acc_accuracy = {}

    # Run inference for each species
    for species, idx in species_mapping.items():
        print(f'Running inference for {species}...')
        df_species = df[df['species'] == idx]
        
        if species in ['pseudomonas_aeruginosa', 'neisseria_gonorrhoeae']:
            model, tokenizer = load_model('/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/outputs/thresh_5e-2_blast_80_pv4/single_model_best', args, device)
        else:
            model, tokenizer = load_model(os.path.join(args.models_dir, species, 'best'), args, device)

        results_df = run_inference(df_species, device, model, tokenizer, args, species)
        acc_preds = get_accession_preds(results_df, idx_to_pheno)

        if args.final_set:
            pass
            # make function to output accession predictions in official CAMDA format here
        else:
            total_acc_true.extend(acc_preds['true_phenotype'])
            total_acc_pred.extend(acc_preds['pred_phenotype'])
            total_seq_true.extend(results_df['phenotype'])
            total_seq_pred.extend(results_df['pred_phenotype_idx'])

            species_acc_accuracy = np.mean(np.array(acc_preds['pred_phenotype']) == 
                                         np.array(acc_preds['true_phenotype']))
            print(f'{species} accession-level accuracy: {species_acc_accuracy:.4f}')
            per_species_acc_accuracy[species] = species_acc_accuracy

        del model
        gc.collect()
        torch.cuda.empty_cache()

        with open(f'{args.run_name}_per_species_acc_accuracy.json', 'w') as f:
            json.dump(per_species_acc_accuracy, f)



    
    # Evaluate
    acc_accuracy = np.mean(np.array(total_acc_pred) == np.array(total_acc_true))
    print(f'Consensus accession-level accuracy (test set): {acc_accuracy:.4f}')
    acc_cm = confusion_matrix(total_acc_true, total_acc_pred)
    print("Confusion matrix (rows: true, cols: pred):\n", acc_cm)
    acc_f1 = f1_score(total_acc_true, total_acc_pred, average='macro')
    print(f"Macro F1 score: {acc_f1:.4f}")

    # --- Sequence-level accuracy and confusion ---
    seq_accuracy = np.mean(np.array(total_seq_pred) == np.array(total_seq_true))
    seq_cm = confusion_matrix(total_seq_true, total_seq_pred)
    print(f"Sequence-level accuracy: {seq_accuracy:.4f}")
    print("Sequence-level confusion matrix (rows: true, cols: pred):\n", seq_cm)
    seq_f1 = f1_score(total_seq_true, total_seq_pred, average='macro')
    print(f"Sequence-level macro F1 score: {seq_f1:.4f}")

    # --- Append summary to CSV ---
    summary_path = args.summary_path
    model_path = args.models_dir
    data_path = args.test_csv
    row = {
        'model_name': args.run_name,
        'model_path': model_path,
        'data_path': data_path,
        'accession_accuracy': float(acc_accuracy),
        'accession_confusion': acc_cm.tolist(),
        'accession_f1': float(acc_f1),
        'sequence_accuracy': float(seq_accuracy),
        'sequence_confusion': seq_cm.tolist(),
        'sequence_f1': float(seq_f1)
    }

    if os.path.exists(summary_path):
        df_summary = pd.read_csv(summary_path, converters={'accession_confusion': ast.literal_eval, 'sequence_confusion': ast.literal_eval})
        df_summary = pd.concat([df_summary, pd.DataFrame([row])], ignore_index=True)
        df_summary.to_csv(summary_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    else:
        pd.DataFrame([row]).to_csv(summary_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

if __name__ == '__main__':
    main()
    





        
