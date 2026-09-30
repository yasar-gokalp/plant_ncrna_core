import itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

print("[1/4] Loading cohort data and reference sequences...")
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

# Assign pair identifiers: first 84 positives match 1:1 with subsequent 84 controls
n_pairs = 84
cohort['pair_id'] = list(range(n_pairs)) + list(range(n_pairs))

# Sequence feature matrices (dinucleotide, 3-mer, 4-mer)
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

X_cov = cohort[['length', 'gc', 'exon_count', 'isoform_count']].values
X_k2 = np.array(mat_2)
X_k3 = np.array(mat_3)
X_k4 = np.array(mat_4)
y = cohort['label'].values
pair_groups = cohort['pair_id'].values

print("\n" + "="*75)
print("1. PAIR-AWARE BENCHMARK: StratifiedGroupKFold (5-Fold)")
print("   (Matched pairs are constrained to the same validation fold)")
print("="*75)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_pair_aware(X_mat, name):
    aucs = []
    for train_idx, test_idx in sgkf.split(X_mat, y, groups=pair_groups):
        rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        rf.fit(X_mat[train_idx], y[train_idx])
        probs = rf.predict_proba(X_mat[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], probs))
    print(f"{name:<35} -> AUROC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    return aucs

cov_scores = evaluate_pair_aware(X_cov, "Covariate-Only (Pair-Aware)")
k2_scores = evaluate_pair_aware(X_k2, "Dinucleotide (16 feat)")
k3_scores = evaluate_pair_aware(X_k3, "3-mer Baseline (64 feat)")
k4_scores = evaluate_pair_aware(X_k4, "4-mer Baseline (256 feat)")

# Label permutation control (within-pair label flipping)
print("\n" + "="*75)
print("2. LABEL PERMUTATION NULL CONTROL (Within-Pair Shuffling, 100 Iterations)")
print("="*75)
perm_aucs = []
np.random.seed(42)
for _ in range(100):
    y_perm = y.copy()
    # Swap labels within each matched pair with 50% probability
    for p in range(n_pairs):
        if np.random.rand() > 0.5:
            idx_pos = p
            idx_neg = p + n_pairs
            y_perm[idx_pos], y_perm[idx_neg] = y_perm[idx_neg], y_perm[idx_pos]
            
    fold_aucs = []
    for train_idx, test_idx in sgkf.split(X_cov, y_perm, groups=pair_groups):
        rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
        rf.fit(X_cov[train_idx], y_perm[train_idx])
        probs = rf.predict_proba(X_cov[test_idx])[:, 1]
        fold_aucs.append(roc_auc_score(y_perm[test_idx], probs))
    perm_aucs.append(np.mean(fold_aucs))

print(f"Permuted Covariate Baseline AUROC (Theoretical Null ~0.50): {np.mean(perm_aucs):.4f} ± {np.std(perm_aucs):.4f}")
print("="*75)
