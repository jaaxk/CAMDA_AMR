import os
import argparse
import pandas as pd
import random
from Bio import SeqIO
from Bio.Seq import Seq

def parse_metadata(metadata_path):
    df = pd.read_csv(metadata_path)
    accession_to_pheno = df.drop_duplicates(subset="accession").set_index("accession")["phenotype"].to_dict()
    return accession_to_pheno

def parse_blast_output(blast_file):
    hits = []
    with open(blast_file) as f:
        for line in f:
            cols = line.strip().split('\t')
            qseqid = cols[0]
            sseqid = cols[1]
            sstart = int(cols[6])
            send = int(cols[7])
            strand = cols[8] if len(cols) > 8 else '+'
            hits.append((qseqid, sseqid, sstart, send, strand))
    return hits

def extract_flanking_sequences(assembly_file, hits, flank=250):
    sequences = []
    seq_dict = SeqIO.to_dict(SeqIO.parse(assembly_file, "fasta"))

    for qseqid, sseqid, sstart, send, strand in hits:
        if sseqid not in seq_dict:
            continue
        contig_seq = seq_dict[sseqid].seq
        seq_len = len(contig_seq)

        mid = (sstart + send) // 2
        start = max(0, mid - flank)
        end = min(seq_len, mid + flank)
        region_seq = contig_seq[start:end]
        if len(region_seq) < 500:
            continue

        if sstart > send or strand == 'minus':
            region_seq = region_seq.reverse_complement()

        sequences.append(str(region_seq))
    return sequences

def main():
    parser = argparse.ArgumentParser(description="Extract 500bp regions from BLAST hits and match phenotypes.")
    parser.add_argument("--metadata", required=True, help="Path to metadata CSV with 'accession' and 'phenotype'")
    parser.add_argument("--blast_dir", default="blast/output", help="Directory with BLAST output files")
    parser.add_argument("--assembly_dir", required=True, help="Directory with all assembly FASTA files")
    parser.add_argument("--output", default="sequences", help="Output CSV path")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle output rows")
    args = parser.parse_args()

    accession_to_pheno = parse_metadata(args.metadata)
    all_data = []
    hits_per_accession = []

    for blast_file in os.listdir(args.blast_dir):
        if not blast_file.endswith(".tsv"):
            continue

        accession = blast_file.split('_')[0]
        phenotype = accession_to_pheno.get(accession)

        if phenotype is None:
            print(f"Skipping {accession} (phenotype not found in metadata)")
            continue

        blast_path = os.path.join(args.blast_dir, blast_file)
        assembly_path = os.path.join(args.assembly_dir, f"{accession}.fasta")
        if not os.path.exists(assembly_path):
            assembly_path = os.path.join(args.assembly_dir, f"{accession}.fa")
        if not os.path.exists(assembly_path):
            print(f"Skipping {accession} (assembly file not found)")
            continue

        hits = parse_blast_output(blast_path)
        sequences = extract_flanking_sequences(assembly_path, hits)

        for seq in sequences:
            all_data.append((seq, phenotype, len(hits), accession))
        
    if args.shuffle:
        print('Shuffling dataset')
        random.shuffle(all_data)

    out_df = pd.DataFrame(all_data, columns=["sequence", "phenotype", "num_hits", "accession"])
    out_df.to_csv(args.output+'_classifier.csv', index=False)
    print(f"Saved {len(out_df)} sequences to {args.output}")

    finetune_df = out_df.drop(columns=["num_hits", "accession"])
    finetune_df.to_csv(args.output+'_finetune.csv', index=False)

if __name__ == "__main__":
    main()

