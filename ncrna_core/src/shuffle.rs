use rand::prelude::*;
use rand::seq::SliceRandom;
use std::thread;
use std::io::{self, BufRead, BufReader, Write};
use std::process::{Command, Stdio};

// Altschul-Erickson dinucleotide shuffle algorithm
pub fn dinuc_shuffle<R: Rng>(seq: &[u8], rng: &mut R) -> Vec<u8> {
    let n = seq.len();
    if n < 3 {
        return seq.to_vec();
    }
    let mut syms: Vec<u8> = Vec::with_capacity(4);
    let mut sym_of = [usize::MAX; 256];
    let mut last = usize::MAX;
    for &b in seq {
        if sym_of[b as usize] == usize::MAX {
            sym_of[b as usize] = syms.len();
            syms.push(b);
        }
        last = sym_of[b as usize];
    }
    let k = syms.len();
    let s: Vec<usize> = seq.iter().map(|&b| sym_of[b as usize]).collect();
    let first = s[0];

    // Build adjacency list for Eulerian path
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); k];
    for w in s.windows(2) {
        adj[w[0]].push(w[1]);
    }

    let last_idx = loop {
        let mut li = vec![usize::MAX; k];
        for v in 0..k {
            if v != last && !adj[v].is_empty() {
                li[v] = rng.gen_range(0..adj[v].len());
            }
        }
        let mut ok = true;
        'check: for v in 0..k {
            if v == last || adj[v].is_empty() {
                continue;
            }
            let mut cur = v;
            let mut steps = 0;
            while cur != last {
                if steps > k || adj[cur].is_empty() {
                    ok = false;
                    break 'check;
                }
                cur = adj[cur][li[cur]];
                steps += 1;
            }
        }
        if ok {
            break li;
        }
    };

    for v in 0..k {
        if adj[v].is_empty() {
            continue;
        }
        if v == last {
            adj[v].shuffle(rng);
        } else {
            let le = adj[v].remove(last_idx[v]);
            adj[v].shuffle(rng);
            adj[v].push(le);
        }
    }

    let mut ptr = vec![0usize; k];
    let mut cur = first;
    let mut out = Vec::with_capacity(n);
    out.push(syms[cur]);
    for _ in 0..n - 1 {
        let nxt = adj[cur][ptr[cur]];
        ptr[cur] += 1;
        out.push(syms[nxt]);
        cur = nxt;
    }
    out
}

#[derive(Debug, Clone)]
pub struct MfeZ {
    pub obs_mfe: f64,
    pub mean_mfe: f64,
    pub std_mfe: f64,
    pub z_score: f64,
    pub percentile: f64,
}

// 64-bit FNV-1a hash for deterministic seed permutation
fn fnv1a(s: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn parse_mfe(line: &str) -> io::Result<f64> {
    let (a, b) = (line.rfind('('), line.rfind(')'));
    match (a, b) {
        (Some(a), Some(b)) if a < b => line[a + 1..b]
            .trim()
            .parse::<f64>()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e)),
        _ => Err(io::Error::new(io::ErrorKind::InvalidData, "Failed to parse MFE")),
    }
}

fn rnafold_batch(seqs: Vec<Vec<u8>>, rnafold: &str) -> io::Result<Vec<f64>> {
    let n = seqs.len();
    let mut child = Command::new(rnafold)
        .arg("--noPS")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;

    let mut stdin = child.stdin.take().unwrap();
    let writer = thread::spawn(move || -> io::Result<()> {
        for s in &seqs {
            stdin.write_all(s)?;
            stdin.write_all(b"\n")?;
        }
        Ok(())
    });

    let stdout = child.stdout.take().unwrap();
    let mut lines = BufReader::new(stdout).lines();
    let mut mfes = Vec::with_capacity(n);
    while let Some(seq_line) = lines.next() {
        let _ = seq_line?;
        match lines.next() {
            Some(Ok(l)) => mfes.push(parse_mfe(&l)?),
            Some(Err(e)) => return Err(e),
            None => break,
        }
    }
    writer.join().map_err(|_| io::Error::new(io::ErrorKind::Other, "Writer thread panicked"))??;
    child.wait()?;

    if mfes.len() != n {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, format!("Expected {} MFE values, received {}", n, mfes.len())));
    }
    Ok(mfes)
}

pub fn mfe_zscore(id: &str, seq: &[u8], n_shuf: usize, seed: u64, rnafold: &str) -> io::Result<MfeZ> {
    let mut rng = StdRng::seed_from_u64(seed ^ fnv1a(id));
    let mut batch: Vec<Vec<u8>> = Vec::with_capacity(n_shuf + 1);
    batch.push(seq.to_vec());
    for _ in 0..n_shuf {
        batch.push(dinuc_shuffle(seq, &mut rng));
    }
    let mfes = rnafold_batch(batch, rnafold)?;
    let obs = mfes[0];
    let null = &mfes[1..];
    let mean = null.iter().sum::<f64>() / null.len() as f64;
    let var = null.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (null.len() - 1).max(1) as f64;
    let sd = var.sqrt();
    let z = if sd > 1e-9 { (obs - mean) / sd } else { f64::NAN };
    let pct = (null.iter().filter(|&&x| x <= obs).count() as f64 + 1.0) / (null.len() as f64 + 1.0);
    Ok(MfeZ {
        obs_mfe: obs,
        mean_mfe: mean,
        std_mfe: sd,
        z_score: z,
        percentile: pct,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn dinuc(s: &[u8]) -> HashMap<(u8, u8), usize> {
        let mut m = HashMap::new();
        for w in s.windows(2) {
            *m.entry((w[0], w[1])).or_insert(0) += 1;
        }
        m
    }

    #[test]
    fn preserves_dinucleotides_and_ends() {
        let seq = b"ACGTGCAACTGACGTGCAACTGA";
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..200 {
            let sh = dinuc_shuffle(seq, &mut rng);
            assert_eq!(sh.len(), seq.len());
            assert_eq!(sh.first(), seq.first());
            assert_eq!(sh.last(), seq.last());
            assert_eq!(dinuc(&sh), dinuc(seq));
        }
    }
}
