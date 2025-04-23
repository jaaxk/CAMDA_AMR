

find . -mindepth 1 -maxdepth 1 -type d ! -exec test -f "{}/contigs.fasta" \; -print | wc

find . -mindepth 1 -maxdepth 1 -type d ! -exec test -f "{}/contigs.fasta" \; -exec rm -r {} +
