import re
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp, mannwhitneyu

print("[1/5] Parsing Araport11.gff3 and transcript FASTA...")

gff_path = 'data/raw/Araport11.gff3'
fasta_path = 'data/raw/araport11_tx.fa'

def clean_id(raw_id):
    tid = raw_id.split()[0]
    tid = tid.replace('transcript:', '').replace('gene:', '').strip()
    return tid

# 1. Load FASTA sequences
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

print(f"Total FASTA transcripts indexed: {len(tx_seqs)}")

# 2. Parse GFF3 annotations
tx_biotype = {}
tx_to_gene = {}
gene_to_txs = defaultdict(set)
exon_counts = defaultdict(int)

with open(gff_path) as f:
    for line in f:
        if line.startswith('#'): continue
        parts = line.strip().split('\t')
        if len(parts) < 9: continue
        ftype, attr = parts[2], parts[8]
        
        m_id = re.search(r'ID=([^;]+)', attr)
        m_parent = re.search(r'Parent=([^;]+)', attr)
        m_bio = re.search(r'biotype=([^;]+)', attr)
        
        tid = clean_id(m_id.group(1)) if m_id else None
        gid = clean_id(m_parent.group(1)) if m_parent else None
        btype = m_bio.group(1) if m_bio else ftype
        
        if ftype not in ['exon', 'CDS', 'five_prime_UTR', 'three_prime_UTR', 'chromosome']:
            if tid:
                tx_biotype[tid] = btype
                if gid:
                    tx_to_gene[tid] = gid
                    gene_to_txs[gid].add(tid)
        elif ftype == 'exon' and gid:
            exon_counts[gid] += 1

# 3. Filter positive candidates
df_v2 = pd.read_csv('data/raw/cohort_276_features_v2.csv')
pos_candidates = [clean_id(x) for x in df_v2['id'].iloc[:138].values]

valid_pos_ids = []
for tid in pos_candidates:
    # Retain annotated ncRNA/lncRNA transcripts
    b = tx_biotype.get(tid, '')
    if ('ncRNA' in b or 'lnc' in b) and tid in tx_seqs:
        valid_pos_ids.append(tid)

# Ensure one representative positive transcript per locus
pos_locus_seen = set()
final_pos_ids = []
for tid in valid_pos_ids:
    gid = tx_to_gene.get(tid, tid.split('.')[0])
    if gid not in pos_locus_seen:
        pos_locus_seen.add(gid)
        final_pos_ids.append(tid)

print(f"Unique positive loci with official ncRNA/lncRNA biotype: {len(final_pos_ids)}")

# 4. Generate candidate negative pool
neg_candidates = []
neg_locus_seen = set()

for tid, btype in tx_biotype.items():
    if ('ncRNA' in btype or 'lnc' in btype) and tid in tx_seqs:
        gid = tx_to_gene.get(tid, tid.split('.')[0])
        if gid not in pos_locus_seen and gid not in neg_locus_seen:
            neg_locus_seen.add(gid)
            neg_candidates.append(tid)

print(f"Independent uncharacterized ncRNA controls available for matching: {len(neg_candidates)}")

# 5. Extract structural and covariate metadata
def extract_meta(tids):
    rows = []
    for tid in tids:
        seq = tx_seqs[tid]
        l = len(seq)
        gc = (seq.count('G') + seq.count('C')) / max(l, 1)
        ex = exon_counts.get(tid, 1)
        gid = tx_to_gene.get(tid, tid.split('.')[0])
        iso = len(gene_to_txs.get(gid, [tid]))
        rows.append({'id': tid, 'length': l, 'gc': gc, 'exon_count': ex, 'isoform_count': iso, 'locus': gid})
    return pd.DataFrame(rows)

pos_df = extract_meta(final_pos_ids)
neg_df = extract_meta(neg_candidates)

# 1:1 Nearest-neighbor matching on standardized covariates
features = ['length', 'gc', 'exon_count', 'isoform_count']
scaler = StandardScaler()

X_neg = scaler.fit_transform(neg_df[features])
X_pos = scaler.transform(pos_df[features])

nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(X_neg)

distances, indices = nn.kneighbors(X_pos)
matched_neg_df = neg_df.iloc[indices.flatten()].copy()

# Assemble matched cohort
pos_df['label'] = 1
matched_neg_df['label'] = 0
cohort = pd.concat([pos_df, matched_neg_df]).reset_index(drop=True)

print("\n" + "="*70)
print("1. COVARIATE BALANCE: POSITIVES VS MATCHED OFFICIAL CONTROLS")
print("="*70)
for feat in features:
    p_vals = cohort.loc[cohort['label']==1, feat]
    n_vals = cohort.loc[cohort['label']==0, feat]
    if feat in ['length', 'gc']:
        stat, p_val = ks_2samp(p_vals, n_vals)
        test_name = "KS test"
    else:
        stat, p_val = mannwhitneyu(p_vals, n_vals)
        test_name = "Mann-Whitney"
    print(f"{feat:<15}: Pos Mean: {p_vals.mean():<8.3f} | Neg Mean: {n_vals.mean():<8.3f} | {test_name} p: {p_val:.4f}")

# 6. Gatekeeper verification test
print("\n" + "="*70)
print("2. GATEKEEPER VALIDATION: COVARIATE-ONLY BASELINE MODEL")
print("="*70)

X_cov = cohort[['length', 'gc', 'exon_count', 'isoform_count']].values
y_cov = cohort['label'].values
groups_cov = cohort['locus'].values

gkf = GroupKFold(n_splits=5)
cov_aucs = []
for train_idx, test_idx in gkf.split(X_cov, y_cov, groups=groups_cov):
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_cov[train_idx], y_cov[train_idx])
    probs = rf.predict_proba(X_cov[test_idx])[:, 1]
    if len(np.unique(y_cov[test_idx])) > 1:
        cov_aucs.append(roc_auc_score(y_cov[test_idx], probs))

gate_auroc = np.mean(cov_aucs)
print(f"Covariate-only Model AUROC: {gate_auroc:.4f} ± {np.std(cov_aucs):.4f}")

if 0.45 <= gate_auroc <= 0.55:
    print("STATUS: Gatekeeper passed (~0.50). Curation and covariate bias successfully neutralized.")
else:
    print(f"STATUS: WARNING. Covariates retain predictive signal (AUROC: {gate_auroc:.4f}).")

cohort.to_csv('data/raw/strictly_matched_cohort.csv', index=False)
print("Matched cohort saved: data/raw/strictly_matched_cohort.csv")
