import itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

print("[1/3] Loading cohort and thermodynamic features...")
cohort = pd.read_csv('data/raw/strictly_matched_cohort.csv')
v2 = pd.read_csv('data/raw/cohort_276_features_v2.csv')
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

# Pair identifier assignment
n_pairs = 84
cohort['pair_id'] = list(range(n_pairs)) + list(range(n_pairs))

# Dinucleotide composition (16 features)
bases = ['A', 'C', 'G', 'T']
kmers_2 = [''.join(p) for p in itertools.product(bases, repeat=2)]
k2_map = {k: i for i, k in enumerate(kmers_2)}

mat_2 = []
for tid in cohort['id']:
    seq = tx_seqs.get(tid, "").replace('U', 'T')
    c2 = np.zeros(16, dtype=float)
    t2 = max(len(seq) - 1, 1)
    for i in range(len(seq) - 1):
        k = seq[i:i+2]
        if k in k2_map: c2[k2_map[k]] += 1.0
    mat_2.append(c2 / t2)

X_k2 = np.array(mat_2)
X_cov = cohort[['length', 'gc', 'exon_count', 'isoform_count']].values
X_comb_cov = np.hstack([X_k2, X_cov])

# Map thermodynamic features from existing v2 table
thermo_cols = ['mfe_obs', 'mfe_density', 'mfe_z', 'min_window_mfe', 'window_mfe_density']
v2_dict = v2.set_index('id')[thermo_cols].to_dict('index')

thermo_mat = []
default_thermo = [v2[col].mean() for col in thermo_cols]
matched_thermo_count = 0

for tid in cohort['id']:
    if tid in v2_dict:
        thermo_mat.append([v2_dict[tid][c] for c in thermo_cols])
        matched_thermo_count += 1
    else:
        # Impute cohort baseline mean if transcript is absent from v2
        thermo_mat.append(default_thermo)

X_thermo = np.array(thermo_mat)
X_k2_thermo = np.hstack([X_k2, X_thermo])

y = cohort['label'].values
pair_groups = cohort['pair_id'].values

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

def run_oof(X_mat):
    oof = np.zeros(len(y))
    scores = []
    for train_idx, test_idx in sgkf.split(X_mat, y, groups=pair_groups):
        rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        rf.fit(X_mat[train_idx], y[train_idx])
        probs = rf.predict_proba(X_mat[test_idx])[:, 1]
        oof[test_idx] = probs
        scores.append(roc_auc_score(y[test_idx], probs))
    return oof, scores

print("[2/3] Fitting models and evaluating out-of-fold predictions...")
oof_cov, s_cov = run_oof(X_cov)
oof_k2, s_k2 = run_oof(X_k2)
oof_comb_cov, s_comb_cov = run_oof(X_comb_cov)
oof_thermo_comb, s_thermo_comb = run_oof(X_k2_thermo)
oof_thermo_only, s_thermo_only = run_oof(X_thermo)

print("\n" + "="*75)
print("1. PAIR-AWARE CROSS-VALIDATION AUROC (84 Matched Pairs)")
print("="*75)
print(f"Residual Covariates-only       : {np.mean(s_cov):.4f} ± {np.std(s_cov):.4f}")
print(f"Thermodynamic-only (5 features): {np.mean(s_thermo_only):.4f} ± {np.std(s_thermo_only):.4f}")
print(f"Dinucleotide (16 features)     : {np.mean(s_k2):.4f} ± {np.std(s_k2):.4f}")
print(f"Dinucleotide + Thermodynamic   : {np.mean(s_thermo_comb):.4f} ± {np.std(s_thermo_comb):.4f}")
print(f"Dinucleotide + Covariates      : {np.mean(s_comb_cov):.4f} ± {np.std(s_comb_cov):.4f}")

print("\n" + "="*75)
print("2. PAIR-AWARE CLUSTER BOOTSTRAP ΔAUROC (1,000 Iterations)")
print("="*75)

np.random.seed(42)
unique_pairs = np.unique(pair_groups)

deltas_thermo_gain = []
deltas_seq_over_cov = []

for _ in range(1000):
    b_pairs = np.random.choice(unique_pairs, size=len(unique_pairs), replace=True)
    b_idx = []
    for p in b_pairs:
        b_idx.extend(np.where(pair_groups == p)[0])
    y_b = y[b_idx]
    if len(np.unique(y_b)) > 1:
        auc_k2 = roc_auc_score(y_b, oof_k2[b_idx])
        auc_k2_th = roc_auc_score(y_b, oof_thermo_comb[b_idx])
        auc_comb_cov = roc_auc_score(y_b, oof_comb_cov[b_idx])
        auc_cov = roc_auc_score(y_b, oof_cov[b_idx])
        
        # Marginal thermodynamic gain: (Dinucleotide + Thermo) - Dinucleotide
        deltas_thermo_gain.append(auc_k2_th - auc_k2)
        # Sequence gain over baseline covariates: (Dinucleotide + Covariates) - Covariates
        deltas_seq_over_cov.append(auc_comb_cov - auc_cov)

ci_th_low, ci_th_high = np.percentile(deltas_thermo_gain, [2.5, 97.5])
p_val_th = np.mean(np.array(deltas_thermo_gain) <= 0)

ci_sc_low, ci_sc_high = np.percentile(deltas_seq_over_cov, [2.5, 97.5])
p_val_sc = np.mean(np.array(deltas_seq_over_cov) <= 0)

print(f"ΔAUROC [(Dinucleotide + Thermo) - Dinucleotide]:")
print(f"  Mean Difference : {np.mean(deltas_thermo_gain):+.4f}")
print(f"  95% Cluster CI  : [{ci_th_low:+.4f}, {ci_th_high:+.4f}]")
print(f"  One-sided p-val : {p_val_th:.4f}")

print(f"\nΔAUROC [(Dinucleotide + Covariates) - Covariates]:")
print(f"  Mean Difference : {np.mean(deltas_seq_over_cov):+.4f}")
print(f"  95% Cluster CI  : [{ci_sc_low:+.4f}, {ci_sc_high:+.4f}]")
print(f"  One-sided p-val : {p_val_sc:.4f}")
print("="*75)
