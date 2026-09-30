import re
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu

print("[1/4] Parsing structural annotations from Araport11.gff3...")
gff_path = 'data/raw/Araport11.gff3'
df_v2 = pd.read_csv('data/raw/cohort_276_features_v2.csv')
target_ids = set(df_v2['id'].values)

exon_counts = defaultdict(int)
tx_to_gene = {}
tx_biotype = {}
gene_to_txs = defaultdict(set)

# Parse GFF3 annotations
with open(gff_path) as f:
    for line in f:
        if line.startswith('#'): continue
        parts = line.strip().split('\t')
        if len(parts) < 9: continue
        
        ftype = parts[2]
        attr = parts[8]
        
        if ftype in ['mRNA', 'lncRNA', 'transcript', 'ncRNA', 'snoRNA', 'snRNA', 'miRNA', 'tRNA', 'rRNA']:
            m_id = re.search(r'ID=([^;]+)', attr)
            m_parent = re.search(r'Parent=([^;]+)', attr)
            if m_id:
                tid = m_id.group(1).replace('transcript:', '')
                tx_biotype[tid] = ftype
                if m_parent:
                    gid = m_parent.group(1).replace('gene:', '')
                    tx_to_gene[tid] = gid
                    gene_to_txs[gid].add(tid)
                    
        elif ftype == 'exon':
            m_parent = re.search(r'Parent=([^;]+)', attr)
            if m_parent:
                parents = m_parent.group(1).replace('transcript:', '').split(',')
                for p in parents:
                    exon_counts[p] += 1

print("[2/4] Extracting structural metadata and covariates for preliminary cohort...")
records = []
for tid in df_v2['id']:
    exons = exon_counts.get(tid, 1)
    gid = tx_to_gene.get(tid, str(tid).split('.')[0])
    isoforms = len(gene_to_txs.get(gid, [tid]))
    btype = tx_biotype.get(tid, 'unknown')
    records.append({
        'id': tid,
        'exon_count': exons,
        'isoform_count': max(isoforms, 1),
        'biotype': btype
    })

cov_df = pd.DataFrame(records)
y = np.array([1]*138 + [0]*138)
cov_df['label'] = y

pos_exons = cov_df.loc[cov_df['label'] == 1, 'exon_count']
neg_exons = cov_df.loc[cov_df['label'] == 0, 'exon_count']
u_stat, p_exon = mannwhitneyu(pos_exons, neg_exons)

pos_iso = cov_df.loc[cov_df['label'] == 1, 'isoform_count']
neg_iso = cov_df.loc[cov_df['label'] == 0, 'isoform_count']
u_stat2, p_iso = mannwhitneyu(pos_iso, neg_iso)

print("\n" + "="*70)
print("1. COVARIATE DISCREPANCY: POSITIVE VS UNMATCHED NEGATIVE CONTROLS")
print("="*70)
print(f"Exon Count    : Pos Mean: {pos_exons.mean():.2f} (Median: {pos_exons.median():.0f}) | Neg Mean: {neg_exons.mean():.2f} (Median: {neg_exons.median():.0f}) | p = {p_exon:.4e}")
print(f"Isoform Count : Pos Mean: {pos_iso.mean():.2f} (Median: {pos_iso.median():.0f}) | Neg Mean: {neg_iso.mean():.2f} (Median: {neg_iso.median():.0f}) | p = {p_iso:.4e}")

print("\nBiotype Distribution:")
print("Curated Positives:")
print(cov_df[cov_df['label']==1]['biotype'].value_counts())
print("\nUnmatched Generic Negatives:")
print(cov_df[cov_df['label']==0]['biotype'].value_counts())

print("\n[3/4] Evaluating ascertainment bias via 10-Fold Locus Group-CV...")
groups = df_v2['id'].apply(lambda x: str(x).split('.')[0] if '.' in str(x) else str(x)).values

# One-hot encode biotype annotations
cov_features = pd.concat([
    cov_df[['exon_count', 'isoform_count']],
    pd.get_dummies(cov_df['biotype'], prefix='bio')
], axis=1).values

dinuc_cols = [c for c in df_v2.columns if c.startswith('f_')]
dinuc_features = df_v2[dinuc_cols].values
combined_features = np.hstack([dinuc_features, cov_features])

gkf = GroupKFold(n_splits=10)

def eval_features(X_mat):
    scores = []
    for train_idx, test_idx in gkf.split(X_mat, y, groups=groups):
        rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        rf.fit(X_mat[train_idx], y[train_idx])
        probs = rf.predict_proba(X_mat[test_idx])[:, 1]
        if len(np.unique(y[test_idx])) > 1:
            scores.append(roc_auc_score(y[test_idx], probs))
    return np.mean(scores), np.std(scores)

auc_cov, sd_cov = eval_features(cov_features)
auc_dinuc, sd_dinuc = eval_features(dinuc_features)
auc_comb, sd_comb = eval_features(combined_features)

print("\n" + "="*70)
print("2. BASELINE COMPARISON: CURATION ARTIFACTS VS SEQUENCE SIGNAL")
print("="*70)
print(f"1. Covariates-only (Exon count + Isoform count + Biotype) : AUROC: {auc_cov:.4f} ± {sd_cov:.4f}")
print(f"2. Dinucleotide frequencies (16 sequence features)       : AUROC: {auc_dinuc:.4f} ± {sd_dinuc:.4f}")
print(f"3. Combined model (Sequence + Metadata covariates)       : AUROC: {auc_comb:.4f} ± {sd_comb:.4f}")
print("="*70)
