# Plant Functional lncRNA Benchmark Dataset & Reference Pipeline (v0.1.0)

This repository provides frozen datasets, pair-aware split allocations, and analysis scripts for:
"Sequence composition distinguishes experimentally characterized plant lncRNAs from annotation-matched ncRNA controls"

## Contents
- `data/strictly_matched_cohort.csv`: 84 experimentally characterized Arabidopsis lncRNAs matched 1:1 against 84 uncharacterized official Araport11 ncRNAs. Matched on length (p=0.93), GC (p=0.99), exon count (p=0.93), and isoform count (p=1.00).
- `data/cohort_84_pairs.fa`: Clean FASTA sequences for all 168 cohort transcripts.
- `scripts/run_pair_aware_benchmark.py`: StratifiedGroupKFold evaluation enforcing twin-pair integrity to eliminate leakage.
- `scripts/run_multi_match_and_discarded_analysis.py`: 20x independent negative draws from the 3,696 candidate pool and profiling of discarded candidates.
