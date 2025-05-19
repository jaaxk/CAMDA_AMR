import os
import csv

FINAL_SETS_DIR = '/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/tune_hyperparams/final_sets'
TRAIN_ACC_FILE = '/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/tune_hyperparams/train_test_dev_accs/train_accs.txt'

# Load master accession list
with open(TRAIN_ACC_FILE, 'r') as f:
    master_accs = set(line.strip() for line in f if line.strip())

print(f"Total accessions in master train list: {len(master_accs)}\n")

# Find all train.csv files in subdirectories of FINAL_SETS_DIR
for subdir in os.listdir(FINAL_SETS_DIR):
    subdir_path = os.path.join(FINAL_SETS_DIR, subdir)
    train_csv = os.path.join(subdir_path, 'train.csv')
    if not os.path.isfile(train_csv):
        continue

    # Read accessions from train.csv
    accessions = set()
    with open(train_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            acc = row.get('accession')
            if acc:
                accessions.add(acc.strip())

    missing = master_accs - accessions
    print(f"Set: {subdir}")
    print(f"  Unique accessions in set: {len(accessions)}")
    print(f"  Missing accessions: {len(missing)}")
    #if missing:
    #    print(f"    Missing: {', '.join(sorted(missing))}")
    print()