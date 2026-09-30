# plant_ncrna_core

**Benchmark and Reference Sequence Profiler for Plant Functional lncRNAs**

This repository provides frozen datasets, pair-aware cross-validation pipelines, and verification models supporting the benchmark:
> *"Sequence composition distinguishes experimentally characterized plant lncRNAs from annotation-matched ncRNA controls"*

## Overview
- **Strictly Matched Cohort**: 84 experimentally characterized *Arabidopsis thaliana* functional lncRNAs evaluated against 84 uncharacterized official Araport11 ncRNAs, matched on length ($p = 0.93$), GC content ($p = 0.99$), exon count ($p = 0.93$), and isoform count ($p = 1.00$).
- **Bias-Controlled Evaluation**: Implements pair-aware group splitting (`StratifiedGroupKFold`) to prevent nearest-neighbor anti-learning artifacts across validation folds.
- **Key Findings**: Isolates sequence composition signal (dinucleotide AUROC ~0.77; $\Delta\text{AUROC} = +0.1680$ over residual covariates, $p < 0.001$) after mitigating database curation bias (which initially inflated covariate models to 0.88).

## Quick Reproduction
```bash
python3 run_pair_aware_benchmark.py
python3 run_multi_match_and_discarded_analysis.py
