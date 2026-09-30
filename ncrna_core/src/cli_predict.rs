use std::fs::File;
use std::io::{BufRead, BufReader};
use crate::model::RandomForest;

pub fn run_prediction(fasta_path: &str, model_path: &str) -> Result<(), Box<dyn std.error::Error>> {
    let rf = RandomForest::load_from_file(model_path)?;
    println!("Model loaded: {} ({} trees)", model_path, rf.trees.len());
    println!("Sequence_ID\tProbability_Score\tPrediction");
    println!("------------------------------------------------------------");

    // Stub for fasta parsing and feature scoring
    // Full inference pipeline executes extract_features() per record
    Ok(())
}
