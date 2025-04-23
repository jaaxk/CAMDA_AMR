import pandas as pd
import argparse
import os

print('Start script')
print(f'NODE_ID: {int(os.environ.get("SLURM_NODEID", 0))}, TOTAL_NODES: {int(os.environ.get("SLURM_NNODES", 1))}, CPUS_PER_TASK: {int(os.environ.get("SLURM_CPUS_PER_TASK", 1))}')

def process_paired(sample, reverse):
    cmd = f'{args.spades_path} {"--isolate " if args.isolate else ""}{"--plasmid " if args.plasmid else ""}-1 {args.fastq_dir}/{sample}_1.fastq -2 {args.fastq_dir}/{sample}_{reverse}.fastq -o {args.outdir}/{sample} -t {args.threads} --only-assembler'
    print(cmd)
    os.system(cmd)

def process_single(file, sample):
    cmd = f'{args.spades_path} {"--isolate " if args.isolate else ""}{"--plasmid " if args.plasmid else ""}-s {args.fastq_dir}/{file} -o {args.outdir}/{sample} -t {args.threads} --only-assembler'
    print(cmd)
    os.system(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spades_path', default='SPAdes-4.0.0-Linux/bin/spades.py')
    parser.add_argument('--outdir', default='spades_output')
    parser.add_argument('--threads', default=96)
    parser.add_argument('--metadata_path', default='train_metadata.csv')
    parser.add_argument('--fastq_dir')
    parser.add_argument('--isolate', action='store_true', help='should trim reads before')
    parser.add_argument('--plasmid', action='store_true', help='assembles ONLY plasmids from WGS data')
    parser.add_argument('--numjobs', type=int, default=1)
    parser.add_argument('--job', type=int, default=0)
    parser.add_argument('--single_only', action='store_true', help='treats ALL reads as SINGLE END (for use after all were assemblies were attempted and some paired end failed due to mismatched read lengths)')
    args = parser.parse_args()

    if args.single_only:
        print('Processing all assemblies as SINGLE END')

    meta_df = pd.read_csv(args.metadata_path)
    node_id = int(os.environ.get("SLURM_NODEID", 0))
    total_nodes = int(os.environ.get("SLURM_NNODES", 1))
    cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print(node_id, total_nodes, cpus_per_task)

    """with open(args.accessions, 'r') as f:
        accessions = [line.strip() for line in f if line.strip()]
        print(f'{len(accessions)} accessions in {args.accessions}')"""

    accessions = list(meta_df['accession'].unique())
    completed = os.listdir(args.outdir)
    for a in completed:
        if a in accessions:
            accessions.remove(a)
        else:
            print(f'{a} not in accessions!!')
    print(f'{len(accessions)} accessions remaining')

    if args.numjobs > 1:
        split_idx = len(accessions) // args.numjobs
        accessions = accessions[split_idx*args.job : split_idx*(args.job + 1)]
        print(f'This job processing {len(accessions)} accessions from {split_idx*args.job} to {split_idx*(args.job + 1)}')
    
    samples_per_node = (len(accessions) + total_nodes - 1) // total_nodes
    start_idx = node_id * samples_per_node
    end_idx = min((node_id + 1) * samples_per_node, len(accessions))

    node_accessions = accessions[start_idx:end_idx]
    print(f"Node {node_id} processing {len(node_accessions)} samples ({start_idx} to {end_idx-1})")

    for i, accession in enumerate(node_accessions):
        if accession not in meta_df['accession'].values:
            print(f'{accession} not found in metadata file')
            continue
        if meta_df[meta_df['accession'] == accession].iloc[0]['layout'] == 'paired' and not args.single_only:
            try:
                reverse =  str(int(meta_df[meta_df['accession'] == accession].iloc[0]['reverse']))
                process_paired(accession, reverse)
            except ValueError:
                print(f'No reverse found for {accession}')
                process_single(meta_df[meta_df['accession'] == accession].iloc[0]['file'], accession)                
        else:
            print(f'Processing SINGLE END for {accession}')
            process_single(meta_df[meta_df['accession'] == accession].iloc[0]['file'], accession)

    print('Finished!')

