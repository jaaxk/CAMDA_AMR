import argparse
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

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


def load_model(model_dir, device):
    BertForSequenceClassification = import_bert_layers(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    model.to(device)
    return model


def preprocess_inputs(df, label_encoder, scaler):
    # Encode species
    species = label_encoder.transform(df['species'])
    # Normalize num_hits
    num_hits = scaler.transform(df['num_hits'].values.reshape(-1, 1)).flatten()
    return species, num_hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help='Path to finetune/outputs/... directory')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV with columns: sequence,num_hits,accession,species,phenotype')
    parser.add_argument('--tokenizer_dir', type=str, required=False, default=None, help='Optional: path to tokenizer')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to output CSV (accession, predicted_phenotype, true_phenotype)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        args.model_dir,
        num_labels=3,
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_dir,
        model_max_length=125,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    # Load data
    df = pd.read_csv(args.input_csv)
    required_cols = ['sequence', 'num_hits', 'accession', 'species', 'phenotype']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Sanity check: all rows for an accession must have the same phenotype
    acc_group = df.groupby('accession')['phenotype'].nunique()
    if (acc_group > 1).any():
        bad = acc_group[acc_group > 1]
        raise ValueError(f"Accessions with inconsistent phenotype: {bad.index.tolist()}")

    # Fit label encoder and scaler on the input data
    label_encoder = LabelEncoder()
    label_encoder.fit(df['species'])
    scaler = MinMaxScaler()
    scaler.fit(df['num_hits'].values.reshape(-1, 1))

    # Prepare mapping for phenotype
    pheno_to_idx = {'Resistant': 0, 'Intermediate': 1, 'Susceptible': 2}
    idx_to_pheno = {v: k for k, v in pheno_to_idx.items()}
    df['phenotype_idx'] = df['phenotype'].map(pheno_to_idx)

    # Inference per row
    batch_size = 32
    pred_phenos = []
    with torch.no_grad():
        for start in tqdm(range(0, len(df), batch_size)):
            batch = df.iloc[start:start+batch_size]
            enc = tokenizer(list(batch['sequence']), padding=True, truncation=True, return_tensors='pt', max_length=125)
            num_hits = scaler.transform(batch['num_hits'].values.reshape(-1, 1)).astype(np.float32)
            species = label_encoder.transform(batch['species'])
            inputs = {
                'input_ids': enc['input_ids'].to(device),
                'attention_mask': enc['attention_mask'].to(device),
                'num_hits': torch.tensor(num_hits, dtype=torch.float32).to(device),
                'species': torch.tensor(species, dtype=torch.long).to(device),
            }
            
            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            pred_phenos.extend([int(x) for x in preds.flatten()])
    df['pred_phenotype_idx'] = pred_phenos
    df['pred_phenotype'] = df['pred_phenotype_idx'].map(idx_to_pheno)

    # Group by accession and take max of predicted phenotype index
    acc_preds = df.groupby('accession').agg({
        'pred_phenotype_idx': 'max',
        'phenotype_idx': 'first',
    }).reset_index()
    acc_preds['pred_phenotype'] = acc_preds['pred_phenotype_idx'].map(idx_to_pheno)
    acc_preds['true_phenotype'] = acc_preds['phenotype_idx'].map(idx_to_pheno)

    # Save output
    acc_preds[['accession', 'pred_phenotype', 'true_phenotype']].to_csv(args.output_csv, index=False)

    # Evaluate
    acc = np.mean(acc_preds['pred_phenotype'] == acc_preds['true_phenotype'])
    print(f'Consensus accession-level accuracy: {acc:.4f}')

if __name__ == '__main__':
    main()
