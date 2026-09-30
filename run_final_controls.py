import re
import itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

print("[1/3] Loading datasets and annotations...")
cohort = pd.read_csv('data/raw/strictly_matched_cohort.csv')
fasta_path = 'data/raw/araport11_tx.fa'
df_v2 = pd.read_csv('data/raw/cohort_276_features_v2.csv')
gff_path = 'data/raw/Araport11.gff3'

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

# Identify the 5 protein-coding transcripts excluded from the final cohort
tx_biotype = {}
with open(gff_path) as f:
    for line in f:
        if line.startswith('#'): continue
        p = line.strip().split('\t')
        if len(p) < 9: continue
        ftype, attr = p[2], p[8]
        m_id = re.search(r'ID=([^;]+)', attr)
        m_bio = re.search(r'biotype=([^;]+)', attr)
        if m_id:
            tid = clean_id(m_id.group(1))
            tx_biotype[tid] = m_bio.group(1) if m_bio else ftype

pos_all = [clean_id(x) for x in df_v2['id'].iloc[:138].values]
coding_pos = [tid for tid in pos_all if tx_biotype.get(tid) == 'protein_coding' and tid in tx_seqs]

# Extract dinucleotide frequency matrices
bases = ['A', 'C', 'G', 'T']
kmers_2 = [''.join(p) for p in itertools.product(bases, repeat=2)]
k2_map = {k: i for i, k in enumerate(kmers_2)}

def extract_k2(tids):
    mat = []
    for tid in tids:
        seq = tx_seqs.get(tid, "").replace('U', 'T')
        c = np.zeros(16, dtype=float)
        t = max(len(seq) - 1, 1)
        for i in range(len(seq) - 1):
            k = seq[i:i+2]
            if k in k2_map: c[k2_map[k]] += 1.0
        mat.append(c / t)
    return np.array(mat)

X_train = extract_k2(cohort['id'])
y_train = cohort['label'].values
pair_groups = list(range(84)) + list(range(84))

# Inspect CD-HIT .clstr output (verify multi-member clusters across pairs)
clstr_path = 'data/raw/cohort_84_c80.clstr'
clusters = {}
curr_cl = None
with open(clstr_path) as f:
    for line in f:
        if line.startswith('>Cluster'):
            curr_cl = line.strip()
            clusters[curr_cl] = []
        else:
            m = re.search(r'>([^.]+?\.\d+)', line)
            if m:
                clusters[curr_cl].append(clean_id(m.group(1)))

multi_clusters = {k: v for k, v in clusters.items() if len(v) > 1}
print(f"CD-HIT clusters with >=80% sequence identity: {len(multi_clusters)}")
for cl, members in multi_clusters.items():
    print(f"  {cl}: {members}")

# Fit model on the 84-pair matched benchmark
rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
rf.fit(X_train, y_train)

# Score excluded protein-coding transcripts as a sanity control
if coding_pos:
    X_coding = extract_k2(coding_pos)
    coding_probs = rf.predict_proba(X_coding)[:, 1]
    print("\n" + "="*70)
    print("CONTROL: PREDICTED PROBABILITIES FOR EXCLUDED PROTEIN-CODING TRANSCRIPTS")
    print("="*70)
    for tid, p in zip(coding_pos, coding_probs):
        print(f"ID: {tid:<15} | Predicted Probability: {p:.4f}")
    print(f"Mean Score: {np.mean(coding_probs):.4f}")
print("="*70)
