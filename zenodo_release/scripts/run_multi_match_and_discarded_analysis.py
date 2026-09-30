import re
import itertools
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

print("[1/3] Araport11 ve pozitif havuz yükleniyor...")
gff_path = 'data/raw/Araport11.gff3'
fasta_path = 'data/raw/araport11_tx.fa'
df_v2 = pd.read_csv('data/raw/cohort_276_features_v2.csv')

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

tx_biotype, tx_to_gene = {}, {}
gene_to_txs = defaultdict(set)
exon_counts = defaultdict(int)

with open(gff_path) as f:
    for line in f:
        if line.startswith('#'): continue
        p = line.strip().split('\t')
        if len(p) < 9: continue
        ftype, attr = p[2], p[8]
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

# Pozitifleri ayır (84 resmi ncRNA vs 54 atılan)
pos_all = [clean_id(x) for x in df_v2['id'].iloc[:138].values]
pos_84, pos_discarded = [], []
pos_locus_seen = set()

for tid in pos_all:
    b = tx_biotype.get(tid, '')
    gid = tx_to_gene.get(tid, tid.split('.')[0])
    if ('ncRNA' in b or 'lnc' in b) and tid in tx_seqs:
        if gid not in pos_locus_seen:
            pos_locus_seen.add(gid)
            pos_84.append(tid)
    else:
        pos_discarded.append(tid)

print(f"84 Tutulan Pozitif, {len(pos_discarded)} Atılan Pozitif (Biyotip uyuşmazlığı / GFF'de bulunamayan).")

# Negatif havuzu (3696 aday)
neg_candidates = []
neg_locus_seen = set()
for tid, btype in tx_biotype.items():
    if ('ncRNA' in btype or 'lnc' in btype) and tid in tx_seqs:
        gid = tx_to_gene.get(tid, tid.split('.')[0])
        if gid not in pos_locus_seen and gid not in neg_locus_seen:
            neg_locus_seen.add(gid)
            neg_candidates.append(tid)

def get_meta(tids):
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

pos_df = get_meta(pos_84)
neg_pool_df = get_meta(neg_candidates)

bases = ['A', 'C', 'G', 'T']
kmers_2 = [''.join(p) for p in itertools.product(bases, repeat=2)]
k2_map = {k: i for i, k in enumerate(kmers_2)}

def get_k2(df_in):
    mat = []
    for tid in df_in['id']:
        seq = tx_seqs.get(tid, "").replace('U', 'T')
        c = np.zeros(16, dtype=float)
        t = max(len(seq) - 1, 1)
        for i in range(len(seq) - 1):
            k = seq[i:i+2]
            if k in k2_map: c[k2_map[k]] += 1.0
        mat.append(c / t)
    return np.array(mat)

# 20 Bağımsız Eşleştirme (k-NN çekilişi gürültüsü ile)
print("\n[2/3] 20 Bağımsız Eşleştirme ve Çift-Farkında CV Simülasyonu...")
features = ['length', 'gc', 'exon_count', 'isoform_count']
scaler = StandardScaler()
X_pool_norm = scaler.fit_transform(neg_pool_df[features])
X_pos_norm = scaler.transform(pos_df[features])

multi_cov_aucs = []
multi_k2_aucs = []

np.random.seed(42)
for draw in range(20):
    # k-NN'de ilk 10 en yakın komşu arasından rastgele 1 tane seçerek 20 farklı eşlenik seti üret
    nn = NearestNeighbors(n_neighbors=10, metric='euclidean')
    nn.fit(X_pool_norm)
    _, indices = nn.kneighbors(X_pos_norm)
    
    selected_neg_indices = [np.random.choice(idx_row) for idx_row in indices]
    neg_draw_df = neg_pool_df.iloc[selected_neg_indices].copy()
    
    pos_df['label'] = 1
    neg_draw_df['label'] = 0
    cur_cohort = pd.concat([pos_df, neg_draw_df]).reset_index(drop=True)
    cur_cohort['pair_id'] = list(range(len(pos_df))) + list(range(len(pos_df)))
    
    X_cov = cur_cohort[features].values
    X_k2 = get_k2(cur_cohort)
    y_c = cur_cohort['label'].values
    p_groups = cur_cohort['pair_id'].values
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=draw)
    
    # Kovaryat
    c_scores, k_scores = [], []
    for train_idx, test_idx in sgkf.split(X_cov, y_c, groups=p_groups):
        rf_c = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        rf_c.fit(X_cov[train_idx], y_c[train_idx])
        c_scores.append(roc_auc_score(y_c[test_idx], rf_c.predict_proba(X_cov[test_idx])[:, 1]))
        
        rf_k = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        rf_k.fit(X_k2[train_idx], y_c[train_idx])
        k_scores.append(roc_auc_score(y_c[test_idx], rf_k.predict_proba(X_k2[test_idx])[:, 1]))
        
    multi_cov_aucs.append(np.mean(c_scores))
    multi_k2_aucs.append(np.mean(k_scores))

print("="*75)
print("1. 20 BAĞIMSIZ EŞLEŞTİRME ÇEKİLİŞİ DAĞILIMI")
print("="*75)
print(f"Kovaryat-Only AUROC (20 Çekiliş Ortalaması) : {np.mean(multi_cov_aucs):.4f} ± {np.std(multi_cov_aucs):.4f} [Aralık: {np.min(multi_cov_aucs):.4f} - {np.max(multi_cov_aucs):.4f}]")
print(f"Dinükleotit AUROC   (20 Çekiliş Ortalaması) : {np.mean(multi_k2_aucs):.4f} ± {np.std(multi_k2_aucs):.4f} [Aralık: {np.min(multi_k2_aucs):.4f} - {np.max(multi_k2_aucs):.4f}]")

# Atılan 54 Pozitifin Dağılımı
print("\n" + "="*75)
print("2. DIŞLANAN 54 POZİTİFİN DETAYI (ASILMA GEREKÇESİ)")
print("="*75)
disc_biotypes = defaultdict(int)
for tid in pos_discarded:
    b = tx_biotype.get(tid, 'GFF3_kaydı_yok / unannotated')
    disc_biotypes[b] += 1
for b, count in disc_biotypes.items():
    print(f"  - {b:<35}: {count} transkript")
print("="*75)
