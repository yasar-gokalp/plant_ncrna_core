use serde::Deserialize;
use std::error::Error;

#[derive(Debug, Deserialize)]
pub struct Node {
    pub feature: usize,
    pub threshold: f64,
    pub left: i32,
    pub right: i32,
    pub value: f64,
    pub is_leaf: bool,
}

#[derive(Debug, Deserialize)]
pub struct Tree {
    pub nodes: Vec<Node>,
}

#[derive(Debug, Deserialize)]
pub struct RandomForest {
    pub feature_names: Vec<String>,
    pub trees: Vec<Tree>,
}

impl RandomForest {
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn Error>> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let rf: RandomForest = serde_json::from_reader(reader)?;
        Ok(rf)
    }

    pub fn predict_proba(&self, features: &[f64]) -> f64 {
        let mut total_prob = 0.0;
        for tree in &self.trees {
            let mut curr = 0usize;
            loop {
                let node = &tree.nodes[curr];
                if node.is_leaf {
                    total_prob += node.value;
                    break;
                }
                if features[node.feature] <= node.threshold {
                    curr = node.left as usize;
                } else {
                    curr = node.right as usize;
                }
            }
        }
        total_prob / (self.trees.len() as f64)
    }
}
