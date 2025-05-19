import os
import argparse
import glob
import pandas as pd
from Bio import SeqIO
import random
import re

def get_file_list(base_dir):
    # Search one subdirectory deep for relevant files
    patterns = ["*.fasta", "*.fna", "*.fa", "*.fq", "*.fastq"]
    files = []
    for subdir in next(os.walk(base_dir))[1]:
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(base_dir, subdir, pattern)))
    return files

def extract_random_subsequences(seq_record, subseq_len, num_seqs):
    seq = str(seq_record.seq)
    if len(seq) < subseq_len:
        return []
    subsequences = []
    for _ in range(num_seqs):
        start = random.randint(0, len(seq) - subseq_len)
        subsequences.append(seq[start:start+subseq_len])
    return subsequences

def main():
    parser = argparse.ArgumentParser(description="Extract random subsequences from fasta/fq files for finetuning dataset.")
    parser.add_argument("--input_dir", required=True, help="Directory containing species subdirectories with fasta/fq files.")
    parser.add_argument("--num_seqs_per_acc", type=int, required=True, help="Number of subsequences per accession.")
    parser.add_argument("--subseq_len", type=int, required=True, help="Length of each subsequence.")
    parser.add_argument("--metadata_csv", required=True, help="Path to metadata CSV file with accession and phenotype.")
    parser.add_argument("--output_csv", required=True, help="Output CSV file path.")
    args = parser.parse_args()

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

    metadata = pd.read_csv(args.metadata_csv)
    acc_to_pheno = dict(zip(metadata['accession'], metadata['phenotype']))

    rows = []
    for subdir in next(os.walk(args.input_dir))[1]:
        species = subdir
        print(f"Processing species: {species}")
        if species not in species_mapping:
            raise ValueError(f"Species '{species}' not in species_mapping dict.")
        species_idx = species_mapping[species]
        subdir_path = os.path.join(args.input_dir, subdir)
        ext = '*.fasta'
        for file_path in glob.glob(os.path.join(subdir_path, ext)):
            accession = os.path.splitext(os.path.basename(file_path))[0]
            if accession not in acc_to_pheno:
                raise ValueError(f"Accession '{accession}' not found in metadata CSV.")
            phenotype = acc_to_pheno[accession]
            # Detect file format for SeqIO
            if file_path.endswith(('.fasta', '.fa', '.fna')):
                fmt = "fasta"
            elif file_path.endswith(('.fq', '.fastq')):
                fmt = "fastq"
            else:
                continue
            seq_records = list(SeqIO.parse(file_path, fmt))
            if not seq_records:
                continue
            random.shuffle(seq_records)  # Randomize order of records
            valid_subseqs = []
            # Try to accumulate up to num_seqs_per_acc valid subsequences
            for rec in seq_records:
                if len(valid_subseqs) >= args.num_seqs_per_acc:
                    break
                subseqs = extract_random_subsequences(rec, args.subseq_len, args.num_seqs_per_acc)
                for subseq in subseqs:
                    if len(valid_subseqs) >= args.num_seqs_per_acc:
                        break
                    # Only allow ACTG (case-insensitive)
                    if re.fullmatch(r'[ACGTacgt]+', subseq):
                        valid_subseqs.append(subseq)
            for subseq in valid_subseqs:
                rows.append({
                    "sequence": subseq,
                    "species_idx": species_idx,
                    "accession": accession,
                    "phenotype": phenotype
                })
            print(f'Rows: {len(rows)}', sep='\r')

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    main()