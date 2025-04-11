import pandas as pd
import argparse
import os
import subprocess

print('Start script')
print(f'NODE_ID: {int(os.environ.get("SLURM_NODEID", 0))}, TOTAL_NODES: {int(os.environ.get("SLURM_NNODES", 1))}, CPUS_PER_TASK: {int(os.environ.get("SLURM_CPUS_PER_TASK", 1))}')

def process_paired(file1, file2, long):
    cmd = f'{'fastplong' if long else 'fastp'} -i {args.indir}/{file1} -I {args.indir}/{file2} -o {args.outdir}/{file1} -O {args.outdir}/{file2} --thread {args.thread}'
    print(cmd)
    r = os.system(cmd)
    check_failed(r, file1)

def process_single(file, long):
    cmd = f'{'fastplong' if long else 'fastp'} -i {args.indir}/{file} -o {args.outdir}/{file} --thread {args.thread}'
    print(cmd)
    r = os.system(cmd)
    check_failed(r, file)

def check_failed(r, file):
    if r == 0:
        return
    else:
        with open(f'{args.outdir}/fastp_logs.txt', 'a') as f:
            f.write(f'{file} failed!\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='fastp')
    parser.add_argument('--indir', default='fastq')
    parser.add_argument('--threads', type=int, default=96)
    parser.add_argument('--metadata_path')
    parser.add_argument('--accessions')
    parser.add_argument('--extension', default='fastq')
    args = parser.parse_args()
    args.thread = max(args.threads, 16)

    meta_df = pd.read_csv(args.metadata_path)
    node_id = int(os.environ.get("SLURM_NODEID", 0))
    total_nodes = int(os.environ.get("SLURM_NNODES", 1))
    cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print(node_id, total_nodes, cpus_per_task)

    """     with open(args.accessions, 'r') as f:
        accessions = [line.strip() for line in f if line.strip()]
        print(f'{len(accessions)} accessions in {args.accessions}') """

    #accessions = list(meta_df['accession'].unique())

    fastq_files = set(os.listdir(args.indir))
    fastp_files = set(os.listdir(args.outdir))
    missing_files = fastq_files - fastp_files
    accessions = set([a.split('_')[0].split('.')[0] for a in missing_files])
    accessions = list(accessions)
    print(f'{len(accessions)} accessions remaining')

    #resume
    """completed = set([a.split('_')[0].split('.')[0] for a in os.listdir(args.outdir)])
    for acc in completed:
        if acc in accessions:
            accessions.remove(acc)
        else:
            print(f'{acc} not in accessions!!')"""
    
    samples_per_node = (len(accessions) + total_nodes - 1) // total_nodes
    start_idx = node_id * samples_per_node
    end_idx = min((node_id + 1) * samples_per_node, len(accessions))

    node_accessions = accessions[start_idx:end_idx]
    print(f"Node {node_id} processing {len(node_accessions)} samples ({start_idx} to {end_idx-1})")

    for i, accession in enumerate(node_accessions):
        if accession not in meta_df['accession'].values:
            print(f'{accession} not found in metadata file')
            continue

        first = meta_df[meta_df['accession'] == accession].iloc[0]['file']

        if meta_df[meta_df['accession'] == accession].iloc[0]['long_short'] == 'short':
            long = False
        else:
            long = True

        if meta_df[meta_df['accession'] == accession].iloc[0]['layout'] == 'paired':
            try:
                reverse =  str(int(meta_df[meta_df['accession'] == accession].iloc[0]['reverse']))
                process_paired(first, accession+'_'+reverse+'.'+args.extension, long)
            except ValueError:
                print(f'No reverse found for {accession}')
                process_single(first, long)
            
        else:
            process_single(first, long)

        if i%500 == 0:
            print(f'Processed {i} samples')

    print('Finished!')
