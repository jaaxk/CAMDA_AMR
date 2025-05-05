import os
import argparse
import pandas as pd
import random
from Bio import SeqIO
from Bio.Seq import Seq


def parse_metadata(camda_df):
    accession_to_pheno = camda_df.drop_duplicates(subset="accession").set_index("accession")["phenotype"].to_dict()
    accession_to_antibiotic = camda_df.drop_duplicates(subset="accession").set_index("accession")["antibiotic"].to_dict()
    accession_to_species = camda_df.drop_duplicates(subset="accession").set_index("accession")["genus_species"].to_dict()

    #Map species and antibiotic to integer labels
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

    antibiotic_mapping = {'GEN': 1, 
        'ERY': 2,
        'CAZ': 3,
        'TET': 4,
        'tetracycline': 4
    }

    label_mapping = {'Resistant': 0, 'Intermediate': 0, 'Susceptible': 1} #treating intermediate as resistant

    accession_to_species = {k: species_mapping[v] for k, v in accession_to_species.items()}
    accession_to_antibiotic = {k: antibiotic_mapping[v] for k, v in accession_to_antibiotic.items()}
    accession_to_pheno = {k: label_mapping[v] for k, v in accession_to_pheno.items()}

    return accession_to_pheno, accession_to_antibiotic, accession_to_species

def normalize_num_hits(num_hits):
    return (float(num_hits) - 0.0) / (496.0 - 0.0)

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

def extract_flanking_sequences(assembly_file, hits, flank, args):
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
        if len(region_seq) < args.subseq_length:
            continue

        if sstart > send or strand == 'minus':
            region_seq = region_seq.reverse_complement()

        sequences.append(str(region_seq))
    return sequences

def main():
    parser = argparse.ArgumentParser(description="Extract x bp regions from BLAST hits and match phenotypes.")
    #parser.add_argument("--metadata", required=True, help="Path to metadata CSV with 'accession' and 'phenotype'")
    parser.add_argument("--blast_dir", default="blast/output", help="Directory with BLAST output files")
    parser.add_argument("--assembly_dir", required=True, help="Directory with all assembly FASTA files")
    parser.add_argument("--output", default="sequences", help="Output CSV path")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle output rows")
    parser.add_argument("--species", required=True, help="Species name to filter accessions (match genus_species column)")
    parser.add_argument("--subseq_length", type=int, help="Length of subsequence to extract")
    parser.add_argument("--min_seqs", type=int, default=55, help="Minimum number of sequences per accession")
    parser.add_argument("--camda_dataset", type=str, default=None, help="Original dataset from CAMDA (for antibiotic, phenotype and measurement value info)" )
    #parser.add_argument("--include_measurement", action="store_true", help="Include measurement value in output")
    args = parser.parse_args()

    df_meta = pd.read_csv(args.camda_dataset)
    df_meta['genus_species'] = df_meta['genus'].str.lower() + '_' + df_meta['species'].str.lower()

    accession_to_pheno, accession_to_antibiotic, accession_to_species = parse_metadata(df_meta)
    # Get all accessions for the requested species
    species_accessions = set(df_meta[df_meta['genus_species'] == args.species]['accession'].unique())

    if not species_accessions:
        print(len(species_accessions))
        raise ValueError(f"ERROR!!!! No accessions found for species {args.species}")

    print(f"Number of accessions for species {args.species}: {len(species_accessions)}")
    all_data = []
    hits_per_accession = []
    accessions_in_all_data = set()

    for blast_file in os.listdir(args.blast_dir):
        if not blast_file.endswith(".tsv"):
            continue

        accession = blast_file.split('_')[0]
        phenotype = accession_to_pheno.get(accession)
        antibiotic = accession_to_antibiotic.get(accession)
        species = accession_to_species.get(accession)

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
        normalized_num_hits = normalize_num_hits(len(hits))
        sequences = extract_flanking_sequences(assembly_path, hits, flank=args.subseq_length//2, args=args)

        for seq in sequences:
            all_data.append((seq, normalized_num_hits, accession, species, antibiotic, phenotype))
            accessions_in_all_data.add(accession)

    # Find accessions for this species not in all_data
    missing_accessions = species_accessions - accessions_in_all_data
    print(f"Number of accessions for species '{args.species}' not in all_data: {len(missing_accessions)}")

    def extract_random_nonoverlapping_sequences(assembly_file, n=args.min_seqs, seq_len=args.subseq_length):
        seqs = []
        try:
            seq_records = list(SeqIO.parse(assembly_file, "fasta"))
        except Exception as e:
            print(f"Error reading {assembly_file}: {e}")
            return []
        for record in seq_records:
            contig_seq = record.seq
            max_start = len(contig_seq) - seq_len
            if max_start < 0:
                continue
            starts = list(range(0, max_start + 1, seq_len))
            random.shuffle(starts)
            count = 0
            for start in starts:
                if count >= n:
                    break
                seqs.append(str(contig_seq[start:start+seq_len]))
                count += 1
            if count >= n:
                break
        return seqs[:n]

    # Add missing accessions
    for accession in missing_accessions:

        phenotype = accession_to_pheno.get(accession)
        antibiotic = accession_to_antibiotic.get(accession)
        species = accession_to_species.get(accession)

        assembly_path = os.path.join(args.assembly_dir, f"{accession}.fasta")
        if not os.path.exists(assembly_path):
            assembly_path = os.path.join(args.assembly_dir, f"{accession}.fa")
        if not os.path.exists(assembly_path):
            print(f"Missing assembly for accession {accession}")
            continue
        seqs = extract_random_nonoverlapping_sequences(assembly_path, n=args.min_seqs, seq_len=args.subseq_length)
        print(f"Accession {accession}: extracted {len(seqs)} non-overlapping {args.subseq_length}bp sequences.")
        if phenotype is None:
            print(f"Warning: phenotype not found for accession {accession}")
            continue

        all_data.extend([(seq, 0, accession, species, antibiotic, phenotype) for seq in seqs])

    # Top up accessions with < min_seqs sequences
    from collections import Counter
    accession_counts = Counter([row[2] for row in all_data])
    for accession in sorted(species_accessions & accessions_in_all_data):

        num_hits = accession_counts.get(accession, 0)
        if 0 <= num_hits < args.min_seqs:

            phenotype = accession_to_pheno.get(accession)
            antibiotic = accession_to_antibiotic.get(accession)
            species = accession_to_species.get(accession)

            n_needed = args.min_seqs - num_hits
            assembly_path = os.path.join(args.assembly_dir, f"{accession}.fasta")
            if not os.path.exists(assembly_path):
                assembly_path = os.path.join(args.assembly_dir, f"{accession}.fa")
            if not os.path.exists(assembly_path):
                print(f"Missing assembly for accession {accession} (for top-up)")
                continue
            seqs = extract_random_nonoverlapping_sequences(assembly_path, n=n_needed, seq_len=args.subseq_length)
            print(f"Topping up accession {accession}: extracted {len(seqs)} additional non-overlapping {args.subseq_length}bp sequences.")
            if phenotype is None:
                print(f"Warning: phenotype not found for accession {accession} (for top-up)")
                continue
            normalized_num_hits = normalize_num_hits(num_hits)
            all_data.extend([(seq, normalized_num_hits, accession, species, antibiotic, phenotype) for seq in seqs])

    if args.shuffle:
        print('Shuffling dataset')
        random.shuffle(all_data)

    out_df = pd.DataFrame(all_data, columns=["sequence", "num_hits", "accession", "species", "antibiotic", "phenotype"])
    out_df.to_csv(args.output+'_classifier.csv', index=False)
    print(f"Saved {len(out_df)} sequences to {args.output}")

    """finetune_df = out_df.drop(columns=["num_hits", "accession"])
    finetune_df = finetune_df.rename(columns={'phenotype':'label'})
    mapping = {'Resistant': 0, 'Intermediate': 1, 'Susceptible': 2}
    finetune_df['label'] = finetune_df['label'].replace(mapping)
    print(finetune_df.head())
    #finetune_df.to_csv(args.output+'_finetune.csv', index=False)
    """

if __name__ == "__main__":
    main()

