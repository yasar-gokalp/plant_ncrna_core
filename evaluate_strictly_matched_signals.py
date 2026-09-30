import itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score

cohort = pd.read_csv('data/raw/strictly_matched_cohort.csv')
fasta_path = 'data/raw/araport11_tx.fa'

def clean_id(raw_id):
    return raw_id.split()[0].replace('transcript:', '').replace('gene:', '').strip()

tx_seqs = {}
curr_id, curr_seq = None, []
with open(fasta_path) as f:
    for line in f:
        line = line.strip()
        if line.startswith('>'):
            if curr_id: tx_seqs[curr_id] = "".join(curr_seq).upper()
            curr_id = clean_id(line[1:])
            curr_seq = []
        else:
            curr_seq.append(line)
    if curr_id: tx_seqs[curr_id] = "".join(curr_seq).upper()

# Extract sequence composition matrices (dinucleotide, 3-mer, 4-mer)
bases = ['A', 'C', 'G', 'T']
kmers_2 = [''.join(p) for p in itertools.product(bases, repeat=2)]
kmers_3 = [''.join(p) for p in itertools.product(bases, repeat=3)]
kmers_4 = [''.join(p) for p in itertools.product(bases, repeat=4)]

k2_map = {k: i for i, k in enumerate(kmers_2)}
k3_map = {k: i for i, k in enumerate(kmers_3)}
k4_map = {k: i for i, k in enumerate(kmers_4)}

mat_2, mat_3, mat_4 = [], [], []

for tid in cohort['id']:
    seq = tx_seqs.get(tid, "").replace('U', 'T')
    
    # 2-mer
    c2 = np.zeros(16, dtype=float)
    t2 = max(len(seq) - 1, 1)
    for i in range(len(seq) - 1):
        k = seq[i:i+2]
        if k in k2_map: c2[k2_map[k]] += 1.0
    mat_2.append(c2 / t2)
    
    # 3-mer
    c3 = np.zeros(64, dtype=float)
    t3 = max(len(seq) - 2, 1)
    for i in range(len(seq) - 2):
        k = seq[i:i+3]
        if k in k3_map: c3[k3_map[k]] += 1.0
    mat_3.append(c3 / t3)
    
    # 4-mer
    c4 = np.zeros(256, dtype=float)
    t4 = max(len(seq) - 3, 1)
    for i in range(len(seq) - 3):
        k = seq[i:i+4]
        if k in k4_map: c4[k4_map[k]] += 1.0
    mat_4.append(c4 / t4)

X_k2 = np.array(mat_2)
X_k3 = np.array(mat_3)
X_k4 = np.array(mat_4)
X_cov = cohort[['length', 'gc', 'exon_count', 'isoform_count']].values
y = cohort['label'].values
groups = cohort['locus'].values

print("======================================================================")
print("1. NEGATIVE CONTROL: COVARIATE-ONLY BASELINE (Stratified 5-Fold)")
print("======================================================================")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cov_aucs = []
for train_idx, test_idx in skf.split(X_cov, y):
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_cov[train_idx], y[train_idx])
    probs = rf.predict_proba(X_cov[test_idx])[:, 1]
    cov_aucs.append(roc_auc_score(y[test_idx], probs))
print(f"Covariate-only AUROC: {np.mean(cov_aucs):.4f} ± {np.std(cov_aucs):.4f}")

print("\n======================================================================")
print("2. SEQUENCE COMPOSITION MODELS (10-Fold Locus-Aware Group-CV)")
print("======================================================================")
gkf = GroupKFold(n_splits=10)

models = {
    "Dinucleotide (16 feat)": X_k2,
    "3-mer Baseline (64 feat)": X_k3,
    "4-mer Baseline (256 feat)": X_k4,
}

for name, X in models.items():
    scores = []
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X[train_idx], y[train_idx])
        probs = rf.predict_proba(X[test_idx])[:, 1]
        if len(np.unique(y[test_idx])) > 1:
            scores.append(roc_auc_score(y[test_idx], probs))
    print(f"{name:<26} -> AUROC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
print("======================================================================")
