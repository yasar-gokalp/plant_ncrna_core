mod model;
mod shuffle;

use needletail::parse_fastx_file;
use rayon::prelude::*;
use serde::Serialize;
use std::env;
use std::error::Error;
use std::fs::File;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::process::{Command, Stdio};
use std::io::{Write, BufRead, BufReader};
use crate::model::RandomForest;

#[derive(Serialize, Clone)]
pub struct TranscriptMetrics {
    pub id: String,
    pub length: usize,
    pub gc_content: f64,
    pub longest_orf: usize,
    pub orf_ratio: f64,
    pub obs_mfe: f64,
    pub mfe_density: f64,
    pub mfe_z: f64,
    pub min_window_mfe: f64,
    pub window_mfe_density: f64,
    pub f_aa: f64, pub f_ac: f64, pub f_ag: f64, pub f_at: f64,
    pub f_ca: f64, pub f_cc: f64, pub f_cg: f64, pub f_ct: f64,
    pub f_ga: f64, pub f_gc: f64, pub f_gg: f64, pub f_gt: f64,
    pub f_ta: f64, pub f_tc: f64, pub f_tg: f64, pub f_tt: f64,
}

impl TranscriptMetrics {
    pub fn to_feature_vec(&self) -> Vec<f64> {
        vec![
            self.length as f64,
            self.gc_content,
            self.longest_orf as f64,
            self.orf_ratio,
            self.obs_mfe,
            self.mfe_density,
            self.mfe_z,
            self.min_window_mfe,
            self.window_mfe_density,
            self.f_aa, self.f_ac, self.f_ag, self.f_at,
            self.f_ca, self.f_cc, self.f_cg, self.f_ct,
            self.f_ga, self.f_gc, self.f_gg, self.f_gt,
            self.f_ta, self.f_tc, self.f_tg, self.f_tt,
        ]
    }
}

fn get_longest_orf(seq: &[u8]) -> usize {
    let mut max_orf = 0;
    let seq_len = seq.len();
    if seq_len < 3 { return 0; }

    for frame in 0..3 {
        let mut in_orf = false;
        let mut current_orf_start = 0;
        let mut i = frame;
        while i + 3 <= seq_len {
            let codon = &seq[i..i+3];
            if !in_orf {
                if codon == b"ATG" || codon == b"AUG" {
                    in_orf = true;
                    current_orf_start = i;
                }
            } else {
                if codon == b"TAA" || codon == b"TAG" || codon == b"TGA" 
                || codon == b"UAA" || codon == b"UAG" || codon == b"UGA" {
                    in_orf = false;
                    let orf_len = i + 3 - current_orf_start;
                    if orf_len > max_orf { max_orf = orf_len; }
                }
            }
            i += 3;
        }
        if in_orf {
            let orf_len = seq_len - current_orf_start;
            if orf_len > max_orf { max_orf = orf_len; }
        }
    }
    max_orf
}

fn run_rnafold_batch(sequences: &[String]) -> Result<Vec<f64>, Box<dyn Error>> {
    let mut child = Command::new("RNAfold")
        .arg("--noPS")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;

    {
        let stdin = child.stdin.as_mut().ok_or("Failed to open stdin")?;
        for (i, seq) in sequences.iter().enumerate() {
            writeln!(stdin, ">{}\n{}", i, seq)?;
        }
    }

    let output = child.wait_with_output()?;
    let reader = BufReader::new(&output.stdout[..]);
    let mut mfes = Vec::with_capacity(sequences.len());

    for line in reader.lines() {
        let l = line?;
        if l.contains(" (") && l.ends_with(')') {
            if let Some(open_paren) = l.rfind(" (") {
                let energy_str = &l[open_paren + 2..l.len() - 1];
                if let Ok(energy) = energy_str.trim().parse::<f64>() {
                    mfes.push(energy);
                }
            }
        }
    }

    if mfes.len() != sequences.len() {
        return Err(format!("RNAfold output mismatch: expected {}, got {}", sequences.len(), mfes.len()).into());
    }

    Ok(mfes)
}

fn calculate_sliding_window_mfe(seq: &str, window_size: usize, step_size: usize) -> Result<f64, Box<dyn Error>> {
    if seq.len() <= window_size {
        let mfes = run_rnafold_batch(&[seq.to_string()])?;
        return Ok(mfes.first().copied().unwrap_or(0.0));
    }

    let mut windows = Vec::new();
    let mut i = 0;
    while i + window_size <= seq.len() {
        windows.push(seq[i..i+window_size].to_string());
        i += step_size;
    }

    let mfes = run_rnafold_batch(&windows)?;
    let min_mfe = mfes.iter().copied().fold(f64::INFINITY, f64::min);
    Ok(min_mfe)
}

fn dump_sliding_window_profile(id: &str, seq: &str, window_size: usize, step_size: usize, out_path: &str) -> Result<(), Box<dyn Error>> {
    let mut windows = Vec::new();
    let mut positions = Vec::new();

    if seq.len() <= window_size {
        windows.push(seq.to_string());
        positions.push((1, seq.len()));
    } else {
        let mut i = 0;
        while i + window_size <= seq.len() {
            windows.push(seq[i..i+window_size].to_string());
            positions.push((i + 1, i + window_size));
            i += step_size;
        }
    }

    let mfes = run_rnafold_batch(&windows)?;

    let mut file = File::create(out_path)?;
    writeln!(file, "transcript_id\tstart\tend\tmidpoint\tmfe")?;
    for ((start, end), mfe) in positions.iter().zip(mfes.iter()) {
        let mid = (start + end) / 2;
        writeln!(file, "{}\t{}\t{}\t{}\t{:.2}", id, start, end, mid, mfe)?;
    }

    Ok(())
}

fn get_dinucleotide_frequencies(seq: &[u8]) -> [f64; 16] {
    let mut counts = [0usize; 16];
    let mut total = 0usize;
    if seq.len() < 2 { return [0.0; 16]; }

    for i in 0..seq.len() - 1 {
        let b1 = match seq[i] { b'A' => 0, b'C' => 1, b'G' => 2, b'T' | b'U' => 3, _ => 4 };
        let b2 = match seq[i+1] { b'A' => 0, b'C' => 1, b'G' => 2, b'T' | b'U' => 3, _ => 4 };
        if b1 < 4 && b2 < 4 {
            counts[b1 * 4 + b2] += 1;
            total += 1;
        }
    }

    let mut freqs = [0.0; 16];
    if total > 0 {
        for (f, &c) in freqs.iter_mut().zip(counts.iter()) {
            *f = c as f64 / total as f64;
        }
    }
    freqs
}

fn process_record(id: &str, seq_bytes: &[u8]) -> Result<TranscriptMetrics, Box<dyn Error>> {
    let length = seq_bytes.len();
    let gc_count = seq_bytes.iter().filter(|&&b| b == b'C' || b == b'G').count();
    let gc_content = if length > 0 { gc_count as f64 / length as f64 } else { 0.0 };

    let longest_orf = get_longest_orf(seq_bytes);
    let orf_ratio = if length > 0 { longest_orf as f64 / length as f64 } else { 0.0 };

    let seq_str = String::from_utf8_lossy(seq_bytes).to_string();
    let obs_mfe = run_rnafold_batch(&[seq_str.clone()])?.first().copied().unwrap_or(0.0);
    let mfe_density = if length > 0 { (obs_mfe / length as f64) * 100.0 } else { 0.0 };

    let min_win_mfe = calculate_sliding_window_mfe(&seq_str, 150, 50)?;
    let win_size_eff = if length < 150 { length } else { 150 };
    let win_mfe_density = if win_size_eff > 0 { (min_win_mfe / win_size_eff as f64) * 100.0 } else { 0.0 };

    let mut rng = rand::thread_rng();
    let mut shuffles = Vec::with_capacity(30);
    for _ in 0..30 {
        let shuf_bytes = shuffle::dinuc_shuffle(seq_bytes, &mut rng);
        shuffles.push(String::from_utf8_lossy(&shuf_bytes).to_string());
    }

    let shuf_mfes = run_rnafold_batch(&shuffles)?;
    let mean_mfe = shuf_mfes.iter().sum::<f64>() / shuf_mfes.len() as f64;
    let var = shuf_mfes.iter().map(|&x| (x - mean_mfe).powi(2)).sum::<f64>() / shuf_mfes.len() as f64;
    let std_mfe = var.sqrt();

    let mfe_z = if std_mfe > 1e-6 { (obs_mfe - mean_mfe) / std_mfe } else { 0.0 };
    let dinuc = get_dinucleotide_frequencies(seq_bytes);

    Ok(TranscriptMetrics {
        id: id.to_string(),
        length,
        gc_content,
        longest_orf,
        orf_ratio,
        obs_mfe,
        mfe_density,
        mfe_z,
        min_window_mfe: min_win_mfe,
        window_mfe_density: win_mfe_density,
        f_aa: dinuc[0], f_ac: dinuc[1], f_ag: dinuc[2], f_at: dinuc[3],
        f_ca: dinuc[4], f_cc: dinuc[5], f_cg: dinuc[6], f_ct: dinuc[7],
        f_ga: dinuc[8], f_gc: dinuc[9], f_gg: dinuc[10], f_gt: dinuc[11],
        f_ta: dinuc[12], f_tc: dinuc[13], f_tg: dinuc[14], f_tt: dinuc[15],
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage:");
        eprintln!("  1) ncrna_core predict <input.fa> <model.json> <output.tsv>");
        eprintln!("  2) ncrna_core profile <input.fa> <transcript_id> <output_profile.tsv>");
        eprintln!("  3) ncrna_core extract <input.fa> <output.csv>");
        std::process::exit(1);
    }

    let mode = &args[1];

    if mode == "profile" {
        if args.len() < 5 {
            eprintln!("Usage: ncrna_core profile <input.fa> <transcript_id> <output_profile.tsv>");
            std::process::exit(1);
        }
        let fasta_path = &args[2];
        let target_id = &args[3];
        let out_path = &args[4];

        let mut reader = parse_fastx_file(fasta_path)?;
        let mut target_seq = None;

        while let Some(r) = reader.next() {
            let rec = r?;
            let id = String::from_utf8_lossy(rec.id()).to_string();
            if id == *target_id || id.starts_with(target_id) {
                target_seq = Some((id, rec.seq().to_ascii_uppercase().to_vec()));
                break;
            }
        }

        match target_seq {
            Some((found_id, seq)) => {
                let seq_str = String::from_utf8_lossy(&seq);
                println!("Found '{}' (length: {} nt). Generating folding profile...", found_id, seq.len());
                dump_sliding_window_profile(&found_id, &seq_str, 150, 25, out_path)?;
                println!("Folding profile saved to: {}", out_path);
            }
            None => {
                eprintln!("ERROR: Sequence identifier '{}' not found in FASTA.", target_id);
                std::process::exit(1);
            }
        }
    } else if mode == "predict" {
        if args.len() < 5 {
            eprintln!("Usage: ncrna_core predict <input.fa> <model.json> <output.tsv>");
            std::process::exit(1);
        }
        let fasta_path = &args[2];
        let model_path = &args[3];
        let out_path = &args[4];

        println!("[1/3] Loading model: {}", model_path);
        let rf = RandomForest::load_from_file(model_path)?;
        println!("      Model loaded ({} trees).", rf.trees.len());

        println!("[2/3] Processing FASTA sequences: {}", fasta_path);
        let mut reader = parse_fastx_file(fasta_path)?;
        let mut records = Vec::new();
        while let Some(r) = reader.next() {
            let rec = r?;
            let id = String::from_utf8_lossy(rec.id()).to_string();
            let seq = rec.seq().to_ascii_uppercase();
            records.push((id, seq.to_vec()));
        }

        let total = records.len();
        println!("      Evaluating {} total sequences...", total);
        let completed = AtomicUsize::new(0);

        let predictions: Vec<(String, usize, f64, f64, f64)> = records.par_iter().map(|(id, seq)| {
            match process_record(id, seq) {
                Ok(metrics) => {
                    let feat = metrics.to_feature_vec();
                    let prob = rf.predict_proba(&feat);
                    let c = completed.fetch_add(1, Ordering::Relaxed) + 1;
                    print!("\r      Progress: {}/{} ({:.1}%)", c, total, (c as f64 / total as f64) * 100.0);
                    let _ = std::io::stdout().flush();
                    (metrics.id, metrics.length, metrics.min_window_mfe, metrics.mfe_z, prob)
                }
                Err(_) => (id.clone(), seq.len(), 0.0, 0.0, -1.0),
            }
        }).collect();
        println!();

        println!("[3/3] Writing predictions to: {}", out_path);
        let mut file = File::create(out_path)?;
        writeln!(file, "transcript_id\tlength\tmin_window_mfe\tmfe_z\tpriority_score\tcategory")?;
        for (id, len, min_mfe, z, prob) in predictions {
            let cat = if prob >= 0.65 {
                "High_Priority"
            } else if prob >= 0.50 {
                "Moderate_Priority"
            } else if prob >= 0.0 {
                "Low_Priority"
            } else {
                "Failed"
            };
            writeln!(file, "{}\t{}\t{:.2}\t{:.2}\t{:.4}\t{}", id, len, min_mfe, z, prob, cat)?;
        }
        println!("Done.");
    } else {
        let fasta_path = if mode == "extract" { &args[2] } else { &args[1] };
        let out_path = if mode == "extract" { &args[3] } else { &args[2] };

        println!("Extracting features: {} -> {}", fasta_path, out_path);
        let mut reader = parse_fastx_file(fasta_path)?;
        let mut records = Vec::new();
        while let Some(r) = reader.next() {
            let rec = r?;
            let id = String::from_utf8_lossy(rec.id()).to_string();
            let seq = rec.seq().to_ascii_uppercase();
            records.push((id, seq.to_vec()));
        }

        let total = records.len();
        let completed = AtomicUsize::new(0);

        let metrics: Vec<TranscriptMetrics> = records.par_iter().filter_map(|(id, seq)| {
            match process_record(id, seq) {
                Ok(m) => {
                    let c = completed.fetch_add(1, Ordering::Relaxed) + 1;
                    print!("\rProgress: {}/{} ({:.1}%)", c, total, (c as f64 / total as f64) * 100.0);
                    let _ = std::io::stdout().flush();
                    Some(m)
                }
                Err(_) => None,
            }
        }).collect();
        println!();

        let mut wtr = csv::Writer::from_path(out_path)?;
        for m in metrics {
            wtr.serialize(m)?;
        }
        println!("Saved to: {}", out_path);
    }

    Ok(())
}
