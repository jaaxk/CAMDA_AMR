import pandas as pd
import argparse
import os
import subprocess
import shutil

print('Start script')

def fastq_to_fasta(fastq_path, fasta_path):
    """Convert fastq to fasta format."""
    with open(fastq_path, 'r') as fq, open(fasta_path, 'w') as fa:
        while True:
            header = fq.readline()
            if not header:
                break
            seq = fq.readline()
            fq.readline()  # plus line
            fq.readline()  # quality line
            fa.write(f'>{header[1:].strip()}\n{seq.strip()}\n')

def process_paired(sample, reverse, retry=False):
    if os.path.isdir(f'{args.outdir}/{sample}'):
        if not os.path.isfile(f'{args.outdir}/{sample}/contigs.fasta'):
            print(f"Dir does not contain contigs.fasta, removing {args.outdir}/{sample}")
            shutil.rmtree(f'{args.outdir}/{sample}')
        else:
            print(f"Dir contains contigs.fasta, skipping {sample}")
            return

    cmd = f'{args.spades_path} {"--isolate " if args.isolate else ""}{"--plasmid " if args.plasmid else ""}-1 {args.fastq_dir}/{sample}_1.fastq -2 {args.fastq_dir}/{sample}_{reverse}.fastq -o {args.outdir}/{sample} -t {args.threads} --only-assembler'
    print(cmd)
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = result.stdout + result.stderr
    print(output)
    if "file not found" in output.lower():
        if not os.listdir(f'{args.outdir}/{sample}'):
            print(f"Removing empty directory {args.outdir}/{sample}")
            shutil.rmtree(f'{args.outdir}/{sample}')
        print(f"'file not found' detected for {sample}, trying single")
        process_single(sample, file_idx=0)
    elif "os return value: -9" in output.lower() or "error" in output.lower():
        if not retry:
            print(f"SPAdes error detected for {sample}, retrying once...")
            process_paired(sample, reverse, retry=True)
        else:
            print(f"SPAdes failed twice for {sample}. Converting fastq to fasta.")
            # Use the first fastq file as input
            fastq_path = f'{args.fastq_dir}/{sample}_1.fastq'
            outdir = f'{args.outdir}/{sample}'
            os.makedirs(outdir, exist_ok=True)
            fasta_path = f'{outdir}/contigs.fasta'
            fastq_to_fasta(fastq_path, fasta_path)
            print(f"Wrote fallback fasta to {fasta_path}")

def process_single(accession, file_idx=0, retry=False):
    files = meta_df[meta_df['accession'] == accession]['file'].tolist()
    sample = accession
    if file_idx >= len(files):
        print(f'**WARNING: No metadata files found in directory for {accession} (tried {file_idx})')
        return
    file = files[file_idx]
    if os.path.isdir(f'{args.outdir}/{sample}'):
        if not os.path.isfile(f'{args.outdir}/{sample}/contigs.fasta'):
            print(f"Dir does not contain contigs.fasta, removing {args.outdir}/{sample}")
            shutil.rmtree(f'{args.outdir}/{sample}')
        else:
            print(f"Dir contains contigs.fasta, skipping {sample}")
            return

    cmd = f'{args.spades_path} {"--isolate " if args.isolate else ""}{"--plasmid " if args.plasmid else ""}-s {args.fastq_dir}/{file} -o {args.outdir}/{sample} -t {args.threads} --only-assembler'
    print(cmd)
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = result.stdout + result.stderr
    print(output)
    if "file not found" in output.lower():
        if not os.listdir(f'{args.outdir}/{sample}'):
            print(f"Removing empty directory {args.outdir}/{sample}")
            shutil.rmtree(f'{args.outdir}/{sample}')
        print(f"'file not found' detected for {accession}, trying next filename")
        process_single(accession, file_idx + 1)
    elif "os return value: -9" in output.lower() or "error" in output.lower():
        if not retry:
            print(f"SPAdes error detected for {accession}, retrying once...")
            process_single(accession, file_idx, retry=True)
        else:
            print(f"SPAdes failed twice for {accession}. Converting fastq to fasta.")
            # Use the current fastq file as input
            fastq_path = f'{args.fastq_dir}/{file}'
            outdir = f'{args.outdir}/{sample}'
            os.makedirs(outdir, exist_ok=True)
            fasta_path = f'{outdir}/contigs.fasta'
            fastq_to_fasta(fastq_path, fasta_path)
            print(f"Wrote fallback fasta to {fasta_path}")


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
    parser.add_argument('--job', type=int, default=None)
    parser.add_argument('--single_only', action='store_true', help='treats ALL reads as SINGLE END (for use after all were assemblies were attempted and some paired end failed due to mismatched read lengths)')
    parser.add_argument('--accessions', type=str, default=None, help='Path to file containing list of accessions to process')
    args = parser.parse_args()

    if args.single_only:
        print('Processing all assemblies as SINGLE END')

    meta_df = pd.read_csv(args.metadata_path)
    node_id = int(os.environ.get("SLURM_NODEID", 0))
    total_nodes = int(os.environ.get("SLURM_NNODES", 1))
    cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    #args.threads = cpus_per_task
    #print(node_id, total_nodes, cpus_per_task)
    if args.job is None:
        args.job = int(os.environ.get("SLURM_PROCID", 0))
        print(f'Job index: {args.job}')

    """with open(args.accessions, 'r') as f:
        accessions = [line.strip() for line in f if line.strip()]
        print(f'{len(accessions)} accessions in {args.accessions}')"""

    #accessions = list(meta_df['accession'].unique())
    with open(args.accessions, 'r') as f:
        accessions = [line.strip() for line in f if line.strip()]
        print(f'{len(accessions)} accessions in {args.accessions}')
    
    #Remove dirs without contigs
    for accession in accessions[:]:
        outdir = f'{args.outdir}/{accession}'
        if os.path.isdir(outdir):
            if not os.path.isfile(f'{outdir}/contigs.fasta'):
                print(f"Removing empty directory {outdir} (no contigs.fasta)")
                try:
                    shutil.rmtree(outdir)
                except FileNotFoundError:
                    print(f"Directory {outdir} not found, skipping removal.")
                except Exception as e:
                    print(f"Warning: Could not remove {outdir}: {e}")


    completed = os.listdir(args.outdir)
    for a in completed:
        if a in accessions:
            accessions.remove(a)
        else:
            print(f'{a} not in accessions!!')
    print(f'{len(accessions)} accessions remaining')

    # Split accessions based on job index and total jobs (numjobs)
    if args.numjobs > 1:
        n = len(accessions)
        jobs = args.numjobs
        job = args.job
        chunk_size = (n + jobs - 1) // jobs  # ceil division
        start = job * chunk_size
        end = min((job + 1) * chunk_size, n)
        accessions = accessions[start:end]
        print(f"This job (index {job} of {jobs}) processing {len(accessions)} accessions: [{start} to {end-1}] out of {n}")
    else:
        print(f"Single job processing all {len(accessions)} accessions.")

    for i, accession in enumerate(accessions):
        if accession not in meta_df['accession'].values:
            print(f'{accession} not found in metadata file')
            continue
        if meta_df[meta_df['accession'] == accession].iloc[0]['layout'] == 'paired' and not args.single_only:
            try:
                reverse =  str(int(meta_df[meta_df['accession'] == accession].iloc[0]['reverse']))
                process_paired(accession, reverse)
            except ValueError:
                print(f'No reverse found for {accession}')
                process_single(accession, file_idx=0)                
        else:
            print(f'Processing SINGLE END for {accession}')
            process_single(accession, file_idx=0)

    print(f'Job {args.job} of {args.numjobs} finished!')
