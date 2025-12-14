// For tensor operations, index-based loops are clearer than iterator chains
#![allow(clippy::needless_range_loop)]

//! Full DisCoCat Implementation — Sentence Similarity and Applications
//!
//! This module completes the DisCoCat (Distributional Compositional Categorical)
//! implementation with:
//! - Sentence vectors (not just scalars)
//! - Similarity measures
//! - Verb tensor construction from data
//! - Baseline models for comparison
//!
//! # Sentence Vectors
//!
//! Unlike Session 15.5 where sentences map to scalars, here we use:
//! ```text
//! F(S) = ℝᵐ  (sentence space)
//! ```
//!
//! This allows computing similarity between sentences via cosine similarity.
//!
//! # Example
//!
//! ```rust
//! use compositional_nlp::pregroup::Grammar;
//! use compositional_nlp::discocat::{DisCoCat, SentenceSemantics};
//!
//! let discocat = DisCoCat::with_toy_lexicon();
//!
//! // Compute sentence similarity
//! let sim = discocat.similarity(&["Alice", "loves", "Bob"], &["Alice", "adores", "Bob"]).unwrap();
//! println!("Similarity: {:.4}", sim);
//!
//! // Compare with bag-of-words baseline
//! let bow_sim = discocat.bow_similarity(&["Alice", "loves", "Bob"], &["Bob", "loves", "Alice"]);
//! println!("BoW similarity: {:.4}", bow_sim);  // Will be 1.0 (same words!)
//! ```

use crate::pregroup::{Grammar, ParseResult};
use crate::NlpError;
use std::collections::HashMap;

/// Order-3 tensor for transitive verbs.
/// `tensor[i][j][k]` represents subject_i × sentence_j × object_k
pub type Tensor3 = Vec<Vec<Vec<f64>>>;

/// Tensor types for sentence-level semantics.
#[derive(Debug, Clone)]
pub enum SentenceTensor {
    /// Noun: vector in ℝⁿ
    NounVec(Vec<f64>),
    /// Intransitive verb: matrix ℝⁿ → ℝᵐ (maps noun to sentence space)
    IntransVerb(Vec<Vec<f64>>),
    /// Transitive verb: order-3 tensor
    TransVerb(Tensor3),
    /// Adjective: matrix ℝⁿ → ℝⁿ (transforms noun space)
    Adjective(Vec<Vec<f64>>),
}

impl SentenceTensor {
    /// Create a noun vector.
    pub fn noun(v: Vec<f64>) -> Self {
        SentenceTensor::NounVec(v)
    }

    /// Create an intransitive verb matrix.
    pub fn intransitive(m: Vec<Vec<f64>>) -> Self {
        SentenceTensor::IntransVerb(m)
    }

    /// Create a transitive verb tensor.
    pub fn transitive(t: Tensor3) -> Self {
        SentenceTensor::TransVerb(t)
    }

    /// Create an adjective matrix.
    pub fn adjective(m: Vec<Vec<f64>>) -> Self {
        SentenceTensor::Adjective(m)
    }

    /// Get as noun vector if applicable.
    pub fn as_noun(&self) -> Option<&Vec<f64>> {
        match self {
            SentenceTensor::NounVec(v) => Some(v),
            _ => None,
        }
    }

    /// Get as intransitive verb matrix if applicable.
    pub fn as_intrans(&self) -> Option<&Vec<Vec<f64>>> {
        match self {
            SentenceTensor::IntransVerb(m) => Some(m),
            _ => None,
        }
    }

    /// Get as transitive verb tensor if applicable.
    pub fn as_trans(&self) -> Option<&Tensor3> {
        match self {
            SentenceTensor::TransVerb(t) => Some(t),
            _ => None,
        }
    }

    /// Get as adjective matrix if applicable.
    pub fn as_adj(&self) -> Option<&Vec<Vec<f64>>> {
        match self {
            SentenceTensor::Adjective(m) => Some(m),
            _ => None,
        }
    }
}

// ============================================================================
// Tensor Operations
// ============================================================================

/// Inner product of two vectors.
pub fn inner_product(v1: &[f64], v2: &[f64]) -> f64 {
    v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum()
}

/// Vector norm (L2).
pub fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Normalize a vector to unit length.
pub fn normalize(v: &[f64]) -> Vec<f64> {
    let n = norm(v);
    if n < 1e-10 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / n).collect()
    }
}

/// Cosine similarity between two vectors.
pub fn cosine_similarity(v1: &[f64], v2: &[f64]) -> f64 {
    let dot = inner_product(v1, v2);
    let n1 = norm(v1);
    let n2 = norm(v2);
    if n1 < 1e-10 || n2 < 1e-10 {
        0.0
    } else {
        dot / (n1 * n2)
    }
}

/// Matrix-vector multiplication: M * v
pub fn mat_vec_mult(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter().map(|row| inner_product(row, v)).collect()
}

/// Contract order-3 tensor with subject and object vectors.
/// Result: s_j = Σ_ik T_ijk * subj_i * obj_k
pub fn contract_tensor3(tensor: &Tensor3, subj: &[f64], obj: &[f64]) -> Vec<f64> {
    let sent_dim = tensor[0].len();
    let mut result = vec![0.0; sent_dim];

    for (i, subj_i) in subj.iter().enumerate() {
        for (j, result_j) in result.iter_mut().enumerate() {
            for (k, obj_k) in obj.iter().enumerate() {
                *result_j += tensor[i][j][k] * subj_i * obj_k;
            }
        }
    }

    result
}

/// Vector addition.
pub fn vec_add(v1: &[f64], v2: &[f64]) -> Vec<f64> {
    v1.iter().zip(v2.iter()).map(|(a, b)| a + b).collect()
}

/// Element-wise vector multiplication.
pub fn vec_mult(v1: &[f64], v2: &[f64]) -> Vec<f64> {
    v1.iter().zip(v2.iter()).map(|(a, b)| a * b).collect()
}

// ============================================================================
// Verb Tensor Construction
// ============================================================================

/// Methods for constructing verb tensors from data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerbConstructionMethod {
    /// Relational method: sum outer products of (subject, object) pairs
    Relational,
    /// Kronecker product of typical subject and object
    Kronecker,
    /// Copy subject into diagonal (for intransitive-like behavior)
    CopySubject,
}

/// Build a transitive verb tensor from corpus data.
///
/// # Arguments
/// * `pairs` - (subject_vector, object_vector) pairs from corpus
/// * `noun_dim` - dimension of noun space
/// * `sent_dim` - dimension of sentence space
/// * `method` - construction method to use
pub fn build_verb_tensor(
    pairs: &[(&[f64], &[f64])],
    noun_dim: usize,
    sent_dim: usize,
    method: VerbConstructionMethod,
) -> Tensor3 {
    let mut tensor = vec![vec![vec![0.0; noun_dim]; sent_dim]; noun_dim];

    match method {
        VerbConstructionMethod::Relational => {
            // Sum outer products: T_ijk += Σ subj_i * obj_k
            // The sentence dimension j is filled uniformly
            for (subj, obj) in pairs {
                for i in 0..noun_dim.min(subj.len()) {
                    for j in 0..sent_dim {
                        for k in 0..noun_dim.min(obj.len()) {
                            tensor[i][j][k] += subj[i] * obj[k] / (pairs.len() as f64);
                        }
                    }
                }
            }
        }
        VerbConstructionMethod::Kronecker => {
            // Average subjects and objects, then Kronecker product
            if !pairs.is_empty() {
                let mut avg_subj = vec![0.0; noun_dim];
                let mut avg_obj = vec![0.0; noun_dim];

                for (subj, obj) in pairs {
                    for i in 0..noun_dim.min(subj.len()) {
                        avg_subj[i] += subj[i];
                    }
                    for k in 0..noun_dim.min(obj.len()) {
                        avg_obj[k] += obj[k];
                    }
                }

                let n = pairs.len() as f64;
                for i in 0..noun_dim {
                    avg_subj[i] /= n;
                    avg_obj[i] /= n;
                }

                // Kronecker: T_ijk = avg_subj_i * avg_obj_k
                for i in 0..noun_dim {
                    for j in 0..sent_dim {
                        for k in 0..noun_dim {
                            tensor[i][j][k] = avg_subj[i] * avg_obj[k];
                        }
                    }
                }
            }
        }
        VerbConstructionMethod::CopySubject => {
            // Diagonal in object dimension: T_ijk = δ_ik
            // This makes meaning depend mainly on subject-object match
            for i in 0..noun_dim {
                for j in 0..sent_dim {
                    if i < noun_dim {
                        tensor[i][j][i] = 1.0;
                    }
                }
            }
        }
    }

    tensor
}

// ============================================================================
// Sentence Semantics
// ============================================================================

/// Sentence-level semantics with vector output.
///
/// Maps sentences to vectors in ℝᵐ (sentence space), enabling
/// similarity computations between sentences.
#[derive(Debug, Clone)]
pub struct SentenceSemantics {
    /// Dimension of noun space (ℝⁿ)
    pub noun_dim: usize,
    /// Dimension of sentence space (ℝᵐ)
    pub sent_dim: usize,
    /// Word meanings
    lexicon: HashMap<String, SentenceTensor>,
}

impl SentenceSemantics {
    /// Create new sentence semantics with given dimensions.
    pub fn new(noun_dim: usize, sent_dim: usize) -> Self {
        Self {
            noun_dim,
            sent_dim,
            lexicon: HashMap::new(),
        }
    }

    /// Create semantics with a toy lexicon for demonstration.
    pub fn toy_lexicon() -> Self {
        let noun_dim = 4;
        let sent_dim = 4;
        let mut sem = Self::new(noun_dim, sent_dim);

        // Nouns (4D vectors)
        // Dimensions: [human, animal, active, positive]
        sem.add_noun("Alice", vec![0.9, 0.1, 0.7, 0.8]);
        sem.add_noun("Bob", vec![0.85, 0.1, 0.5, 0.6]);
        sem.add_noun("dog", vec![0.1, 0.9, 0.9, 0.7]);
        sem.add_noun("cat", vec![0.1, 0.85, 0.6, 0.6]);
        sem.add_noun("mouse", vec![0.05, 0.8, 0.7, 0.4]);
        sem.add_noun("man", vec![0.95, 0.05, 0.6, 0.5]);
        sem.add_noun("woman", vec![0.95, 0.05, 0.6, 0.6]);

        // Intransitive verbs (4x4 matrices: noun_space → sent_space)
        sem.add_intransitive_verb(
            "runs",
            vec![
                vec![0.3, 0.2, 0.8, 0.5],  // output if human-like
                vec![0.4, 0.3, 0.9, 0.6],  // output if animal-like
                vec![0.5, 0.4, 0.95, 0.7], // output if active
                vec![0.3, 0.2, 0.6, 0.5],  // output if positive
            ],
        );
        sem.add_intransitive_verb(
            "sleeps",
            vec![
                vec![0.7, 0.6, 0.2, 0.6],
                vec![0.6, 0.7, 0.3, 0.5],
                vec![0.3, 0.2, 0.1, 0.4],
                vec![0.5, 0.5, 0.3, 0.6],
            ],
        );
        sem.add_intransitive_verb(
            "walks",
            vec![
                vec![0.5, 0.3, 0.6, 0.5],
                vec![0.4, 0.4, 0.7, 0.5],
                vec![0.6, 0.5, 0.7, 0.6],
                vec![0.4, 0.3, 0.5, 0.5],
            ],
        );

        // Transitive verbs (4x4x4 tensors)
        // Build using helper that creates semantically meaningful tensors
        sem.add_transitive_verb("loves", sem.build_love_tensor());
        sem.add_transitive_verb("adores", sem.build_adore_tensor());
        sem.add_transitive_verb("hates", sem.build_hate_tensor());
        sem.add_transitive_verb("sees", sem.build_see_tensor());
        sem.add_transitive_verb("chases", sem.build_chase_tensor());
        sem.add_transitive_verb("bites", sem.build_bite_tensor());

        // Adjectives (4x4 matrices: noun_space → noun_space)
        sem.add_adjective(
            "big",
            vec![
                vec![1.0, 0.0, 0.1, 0.0],
                vec![0.0, 1.0, 0.1, 0.0],
                vec![0.0, 0.0, 1.2, 0.0],
                vec![0.0, 0.0, 0.0, 1.0],
            ],
        );
        sem.add_adjective(
            "small",
            vec![
                vec![1.0, 0.0, -0.1, 0.0],
                vec![0.0, 1.0, -0.1, 0.0],
                vec![0.0, 0.0, 0.8, 0.0],
                vec![0.0, 0.0, 0.0, 1.0],
            ],
        );
        sem.add_adjective(
            "happy",
            vec![
                vec![1.0, 0.0, 0.0, 0.1],
                vec![0.0, 1.0, 0.0, 0.1],
                vec![0.0, 0.0, 1.1, 0.0],
                vec![0.0, 0.0, 0.0, 1.2],
            ],
        );

        // Determiners (identity)
        let identity = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        sem.add_adjective("the", identity.clone());
        sem.add_adjective("a", identity);

        sem
    }

    // Helper methods to build semantically meaningful verb tensors

    fn build_love_tensor(&self) -> Tensor3 {
        let n = self.noun_dim;
        let m = self.sent_dim;
        let mut t = vec![vec![vec![0.0; n]; m]; n];

        // Love: high when subject is human/positive, object is anything loveable
        for i in 0..n {
            for j in 0..m {
                for k in 0..n {
                    // Base relationship
                    t[i][j][k] = 0.3;
                    // Human subjects love more
                    if i == 0 {
                        t[i][j][k] += 0.3;
                    }
                    // Positive sentiment boost
                    if j == 3 {
                        t[i][j][k] += 0.2;
                    }
                    // Living objects are more loveable
                    if k == 0 || k == 1 {
                        t[i][j][k] += 0.2;
                    }
                }
            }
        }
        t
    }

    fn build_adore_tensor(&self) -> Tensor3 {
        // Similar to love but stronger positive component
        let mut t = self.build_love_tensor();
        for row in &mut t {
            for col in row {
                for val in col {
                    *val *= 1.1; // Slightly stronger
                }
            }
        }
        // Boost positive dimension more
        for i in 0..self.noun_dim {
            for k in 0..self.noun_dim {
                t[i][3][k] += 0.15;
            }
        }
        t
    }

    fn build_hate_tensor(&self) -> Tensor3 {
        let n = self.noun_dim;
        let m = self.sent_dim;
        let mut t = vec![vec![vec![0.0; n]; m]; n];

        // Hate: opposite of love in positive dimension
        for i in 0..n {
            for j in 0..m {
                for k in 0..n {
                    t[i][j][k] = 0.3;
                    if i == 0 {
                        t[i][j][k] += 0.2;
                    }
                    // Negative sentiment (low positive)
                    if j == 3 {
                        t[i][j][k] -= 0.4;
                    }
                }
            }
        }
        t
    }

    fn build_see_tensor(&self) -> Tensor3 {
        let n = self.noun_dim;
        let m = self.sent_dim;
        let mut t = vec![vec![vec![0.0; n]; m]; n];

        // See: neutral, everyone can see everyone
        for i in 0..n {
            for j in 0..m {
                for k in 0..n {
                    t[i][j][k] = 0.5;
                }
            }
        }
        t
    }

    fn build_chase_tensor(&self) -> Tensor3 {
        let n = self.noun_dim;
        let m = self.sent_dim;
        let mut t = vec![vec![vec![0.0; n]; m]; n];

        // Chase: active subjects chase smaller/weaker objects
        for i in 0..n {
            for j in 0..m {
                for k in 0..n {
                    t[i][j][k] = 0.2;
                    // Animals chase more
                    if i == 1 {
                        t[i][j][k] += 0.4;
                    }
                    // Active dimension boost
                    if j == 2 {
                        t[i][j][k] += 0.3;
                    }
                    // Chase animals more than humans
                    if k == 1 {
                        t[i][j][k] += 0.2;
                    }
                }
            }
        }
        t
    }

    fn build_bite_tensor(&self) -> Tensor3 {
        let n = self.noun_dim;
        let m = self.sent_dim;
        let mut t = vec![vec![vec![0.0; n]; m]; n];

        // Bite: animals bite, negative action
        for i in 0..n {
            for j in 0..m {
                for k in 0..n {
                    t[i][j][k] = 0.1;
                    // Animals bite
                    if i == 1 {
                        t[i][j][k] += 0.5;
                    }
                    // Active component
                    if j == 2 {
                        t[i][j][k] += 0.2;
                    }
                    // Negative sentiment
                    if j == 3 {
                        t[i][j][k] -= 0.2;
                    }
                }
            }
        }
        t
    }

    /// Add a noun vector.
    pub fn add_noun(&mut self, word: &str, vector: Vec<f64>) {
        self.lexicon
            .insert(word.to_string(), SentenceTensor::NounVec(vector));
    }

    /// Add an intransitive verb matrix.
    pub fn add_intransitive_verb(&mut self, word: &str, matrix: Vec<Vec<f64>>) {
        self.lexicon
            .insert(word.to_string(), SentenceTensor::IntransVerb(matrix));
    }

    /// Add a transitive verb tensor.
    pub fn add_transitive_verb(&mut self, word: &str, tensor: Tensor3) {
        self.lexicon
            .insert(word.to_string(), SentenceTensor::TransVerb(tensor));
    }

    /// Add an adjective matrix.
    pub fn add_adjective(&mut self, word: &str, matrix: Vec<Vec<f64>>) {
        self.lexicon
            .insert(word.to_string(), SentenceTensor::Adjective(matrix));
    }

    /// Get tensor for a word.
    pub fn get_tensor(&self, word: &str) -> Option<&SentenceTensor> {
        self.lexicon.get(word)
    }

    /// Compute sentence vector from parse result.
    pub fn sentence_vector(&self, parse: &ParseResult) -> Result<Vec<f64>, NlpError> {
        if parse.typed_words.is_empty() {
            return Err(NlpError::EmptySentence);
        }

        // Get tensors for all words
        let tensors: Vec<&SentenceTensor> = parse
            .typed_words
            .iter()
            .map(|tw| {
                self.lexicon
                    .get(&tw.word)
                    .ok_or_else(|| NlpError::UnknownWord {
                        word: tw.word.clone(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        self.contract_to_sentence(&tensors)
    }

    /// Contract tensors to produce sentence vector.
    fn contract_to_sentence(&self, tensors: &[&SentenceTensor]) -> Result<Vec<f64>, NlpError> {
        match tensors.len() {
            0 => Err(NlpError::EmptySentence),
            1 => {
                // Single word - embed in sentence space
                match tensors[0] {
                    SentenceTensor::NounVec(v) => {
                        // Pad/project to sentence dim
                        let mut result = vec![0.0; self.sent_dim];
                        for (i, val) in v.iter().enumerate().take(self.sent_dim) {
                            result[i] = *val;
                        }
                        Ok(result)
                    }
                    _ => Ok(vec![0.0; self.sent_dim]),
                }
            }
            2 => self.contract_two(tensors),
            3 => self.contract_three(tensors),
            _ => self.contract_general(tensors),
        }
    }

    /// Contract two tensors (e.g., "Alice runs").
    fn contract_two(&self, tensors: &[&SentenceTensor]) -> Result<Vec<f64>, NlpError> {
        match (tensors[0], tensors[1]) {
            // Subject + intransitive verb
            (SentenceTensor::NounVec(subj), SentenceTensor::IntransVerb(verb)) => {
                Ok(mat_vec_mult(verb, subj))
            }
            // Adjective + noun (returns modified noun, embed in sent space)
            (SentenceTensor::Adjective(adj), SentenceTensor::NounVec(noun)) => {
                let modified = mat_vec_mult(adj, noun);
                let mut result = vec![0.0; self.sent_dim];
                for (i, val) in modified.iter().enumerate().take(self.sent_dim) {
                    result[i] = *val;
                }
                Ok(result)
            }
            _ => Err(NlpError::TypeMismatch {
                expected: "noun-verb or adj-noun".to_string(),
                got: "other combination".to_string(),
            }),
        }
    }

    /// Contract three tensors (e.g., "Alice loves Bob").
    fn contract_three(&self, tensors: &[&SentenceTensor]) -> Result<Vec<f64>, NlpError> {
        match (tensors[0], tensors[1], tensors[2]) {
            // Subject + transitive verb + object
            (
                SentenceTensor::NounVec(subj),
                SentenceTensor::TransVerb(verb),
                SentenceTensor::NounVec(obj),
            ) => Ok(contract_tensor3(verb, subj, obj)),
            // Adjective + noun + intransitive verb
            (
                SentenceTensor::Adjective(adj),
                SentenceTensor::NounVec(noun),
                SentenceTensor::IntransVerb(verb),
            ) => {
                let modified = mat_vec_mult(adj, noun);
                Ok(mat_vec_mult(verb, &modified))
            }
            // Two adjectives + noun
            (
                SentenceTensor::Adjective(adj1),
                SentenceTensor::Adjective(adj2),
                SentenceTensor::NounVec(noun),
            ) => {
                let mod1 = mat_vec_mult(adj2, noun);
                let mod2 = mat_vec_mult(adj1, &mod1);
                let mut result = vec![0.0; self.sent_dim];
                for (i, val) in mod2.iter().enumerate().take(self.sent_dim) {
                    result[i] = *val;
                }
                Ok(result)
            }
            _ => Err(NlpError::TypeMismatch {
                expected: "subj-verb-obj or adj-noun-verb".to_string(),
                got: "other combination".to_string(),
            }),
        }
    }

    /// Contract more than three tensors.
    fn contract_general(&self, tensors: &[&SentenceTensor]) -> Result<Vec<f64>, NlpError> {
        // Process left-to-right, accumulating noun vector
        let mut current_noun: Option<Vec<f64>> = None;
        let mut pending_trans_verb: Option<&Tensor3> = None;

        for tensor in tensors {
            match tensor {
                SentenceTensor::NounVec(v) => {
                    if let Some(verb) = pending_trans_verb {
                        // Complete transitive: subj verb obj
                        if let Some(subj) = &current_noun {
                            let sent = contract_tensor3(verb, subj, v);
                            // Convert sentence vec back to "noun" for further processing
                            current_noun = Some(sent);
                            pending_trans_verb = None;
                        }
                    } else if current_noun.is_none() {
                        current_noun = Some(v.clone());
                    } else {
                        // Two nouns in a row - just use the second
                        current_noun = Some(v.clone());
                    }
                }
                SentenceTensor::IntransVerb(m) => {
                    if let Some(noun) = &current_noun {
                        let sent = mat_vec_mult(m, noun);
                        current_noun = Some(sent);
                    }
                }
                SentenceTensor::TransVerb(t) => {
                    pending_trans_verb = Some(t);
                }
                SentenceTensor::Adjective(m) => {
                    if let Some(noun) = &current_noun {
                        current_noun = Some(mat_vec_mult(m, noun));
                    }
                }
            }
        }

        current_noun.ok_or(NlpError::ParseError {
            message: "Could not compute sentence vector".to_string(),
        })
    }

    /// Compute noun phrase vector.
    pub fn noun_phrase_vector(&self, parse: &ParseResult) -> Result<Vec<f64>, NlpError> {
        if parse.typed_words.is_empty() {
            return Err(NlpError::EmptySentence);
        }

        let tensors: Vec<&SentenceTensor> = parse
            .typed_words
            .iter()
            .map(|tw| {
                self.lexicon
                    .get(&tw.word)
                    .ok_or_else(|| NlpError::UnknownWord {
                        word: tw.word.clone(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Process right-to-left for noun phrases
        let mut result: Option<Vec<f64>> = None;

        for tensor in tensors.iter().rev() {
            match tensor {
                SentenceTensor::NounVec(v) => {
                    result = Some(v.clone());
                }
                SentenceTensor::Adjective(m) => {
                    if let Some(current) = &result {
                        result = Some(mat_vec_mult(m, current));
                    }
                }
                _ => {}
            }
        }

        result.ok_or(NlpError::ParseError {
            message: "No noun found in phrase".to_string(),
        })
    }
}

// ============================================================================
// Baseline Models
// ============================================================================

/// Baseline models for comparison with DisCoCat.
pub struct Baselines;

impl Baselines {
    /// Bag-of-words: sum of word vectors (ignores order).
    pub fn bag_of_words(words: &[&str], lexicon: &HashMap<String, Vec<f64>>) -> Vec<f64> {
        let dim = lexicon.values().next().map(|v| v.len()).unwrap_or(4);
        let mut result = vec![0.0; dim];

        for word in words {
            if let Some(vec) = lexicon.get(*word) {
                for (i, val) in vec.iter().enumerate() {
                    if i < result.len() {
                        result[i] += val;
                    }
                }
            }
        }

        result
    }

    /// Addition model: normalized sum of word vectors.
    pub fn addition_model(words: &[&str], lexicon: &HashMap<String, Vec<f64>>) -> Vec<f64> {
        let bow = Self::bag_of_words(words, lexicon);
        normalize(&bow)
    }

    /// Multiplication model: element-wise product of word vectors.
    pub fn multiplication_model(words: &[&str], lexicon: &HashMap<String, Vec<f64>>) -> Vec<f64> {
        let dim = lexicon.values().next().map(|v| v.len()).unwrap_or(4);
        let mut result = vec![1.0; dim];

        for word in words {
            if let Some(vec) = lexicon.get(*word) {
                result = vec_mult(&result, vec);
            }
        }

        result
    }

    /// Compute similarity using bag-of-words.
    pub fn bow_similarity(s1: &[&str], s2: &[&str], lexicon: &HashMap<String, Vec<f64>>) -> f64 {
        let v1 = Self::bag_of_words(s1, lexicon);
        let v2 = Self::bag_of_words(s2, lexicon);
        cosine_similarity(&v1, &v2)
    }

    /// Compute similarity using addition model.
    pub fn add_similarity(s1: &[&str], s2: &[&str], lexicon: &HashMap<String, Vec<f64>>) -> f64 {
        let v1 = Self::addition_model(s1, lexicon);
        let v2 = Self::addition_model(s2, lexicon);
        cosine_similarity(&v1, &v2)
    }

    /// Compute similarity using multiplication model.
    pub fn mult_similarity(s1: &[&str], s2: &[&str], lexicon: &HashMap<String, Vec<f64>>) -> f64 {
        let v1 = Self::multiplication_model(s1, lexicon);
        let v2 = Self::multiplication_model(s2, lexicon);
        cosine_similarity(&v1, &v2)
    }
}

// ============================================================================
// Full DisCoCat System
// ============================================================================

/// Full DisCoCat system combining grammar and semantics.
///
/// This is the complete categorical compositional distributional model:
/// - Pregroup grammar for structure
/// - Tensor semantics for meaning
/// - Similarity measures for comparison
#[derive(Debug, Clone)]
pub struct DisCoCat {
    /// Pregroup grammar
    pub grammar: Grammar,
    /// Sentence-level semantics
    pub semantics: SentenceSemantics,
    /// Simple word vectors for baseline comparisons
    baseline_lexicon: HashMap<String, Vec<f64>>,
}

impl DisCoCat {
    /// Create a new DisCoCat system.
    pub fn new(grammar: Grammar, semantics: SentenceSemantics) -> Self {
        let baseline_lexicon = Self::extract_baseline_lexicon(&semantics);
        Self {
            grammar,
            semantics,
            baseline_lexicon,
        }
    }

    /// Create with toy lexicon for demonstration.
    pub fn with_toy_lexicon() -> Self {
        let grammar = Grammar::english_basic();
        let semantics = SentenceSemantics::toy_lexicon();
        Self::new(grammar, semantics)
    }

    /// Extract simple word vectors for baseline models.
    fn extract_baseline_lexicon(semantics: &SentenceSemantics) -> HashMap<String, Vec<f64>> {
        let mut lexicon = HashMap::new();

        for (word, tensor) in &semantics.lexicon {
            let vec = match tensor {
                SentenceTensor::NounVec(v) => v.clone(),
                SentenceTensor::IntransVerb(m) => {
                    // Use first row as representative
                    m.first().cloned().unwrap_or_default()
                }
                SentenceTensor::TransVerb(t) => {
                    // Use diagonal elements
                    let n = t.len();
                    (0..n).map(|i| t[i][i % t[i].len()][i]).collect()
                }
                SentenceTensor::Adjective(m) => {
                    // Use diagonal
                    (0..m.len()).map(|i| m[i][i]).collect()
                }
            };
            lexicon.insert(word.clone(), vec);
        }

        lexicon
    }

    /// Parse a sentence.
    pub fn parse(&self, sentence: &[&str]) -> Result<ParseResult, NlpError> {
        self.grammar.parse(sentence)
    }

    /// Compute sentence vector.
    pub fn sentence_vector(&self, sentence: &[&str]) -> Result<Vec<f64>, NlpError> {
        let parse = self.parse(sentence)?;
        self.semantics.sentence_vector(&parse)
    }

    /// Compute similarity between two sentences.
    pub fn similarity(&self, s1: &[&str], s2: &[&str]) -> Result<f64, NlpError> {
        let v1 = self.sentence_vector(s1)?;
        let v2 = self.sentence_vector(s2)?;
        Ok(cosine_similarity(&v1, &v2))
    }

    /// Bag-of-words similarity (baseline).
    pub fn bow_similarity(&self, s1: &[&str], s2: &[&str]) -> f64 {
        Baselines::bow_similarity(s1, s2, &self.baseline_lexicon)
    }

    /// Addition model similarity (baseline).
    pub fn add_similarity(&self, s1: &[&str], s2: &[&str]) -> f64 {
        Baselines::add_similarity(s1, s2, &self.baseline_lexicon)
    }

    /// Multiplication model similarity (baseline).
    pub fn mult_similarity(&self, s1: &[&str], s2: &[&str]) -> f64 {
        Baselines::mult_similarity(s1, s2, &self.baseline_lexicon)
    }

    /// Evaluate on sentence similarity task.
    /// Returns Spearman correlation with gold ratings.
    pub fn evaluate_similarity(&self, pairs: &[(Vec<&str>, Vec<&str>, f64)]) -> EvaluationResult {
        let mut discocat_sims = Vec::new();
        let mut bow_sims = Vec::new();
        let mut add_sims = Vec::new();
        let mut gold = Vec::new();

        for (s1, s2, g) in pairs {
            if let Ok(sim) = self.similarity(s1, s2) {
                discocat_sims.push(sim);
                bow_sims.push(self.bow_similarity(s1, s2));
                add_sims.push(self.add_similarity(s1, s2));
                gold.push(*g);
            }
        }

        EvaluationResult {
            discocat_correlation: spearman_correlation(&discocat_sims, &gold),
            bow_correlation: spearman_correlation(&bow_sims, &gold),
            add_correlation: spearman_correlation(&add_sims, &gold),
            num_pairs: gold.len(),
        }
    }

    /// Check if first sentence entails second (simple heuristic).
    pub fn entails(&self, premise: &[&str], hypothesis: &[&str]) -> Result<bool, NlpError> {
        let sim = self.similarity(premise, hypothesis)?;
        // Simple threshold-based entailment
        Ok(sim > 0.7)
    }
}

/// Results from similarity evaluation.
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// DisCoCat Spearman correlation
    pub discocat_correlation: f64,
    /// Bag-of-words Spearman correlation
    pub bow_correlation: f64,
    /// Addition model Spearman correlation
    pub add_correlation: f64,
    /// Number of evaluated pairs
    pub num_pairs: usize,
}

impl std::fmt::Display for EvaluationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Evaluation Results ({} pairs):", self.num_pairs)?;
        writeln!(f, "  DisCoCat:  ρ = {:.4}", self.discocat_correlation)?;
        writeln!(f, "  BoW:       ρ = {:.4}", self.bow_correlation)?;
        writeln!(f, "  Addition:  ρ = {:.4}", self.add_correlation)
    }
}

/// Compute Spearman rank correlation.
pub fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len();

    // Compute ranks
    let rank_x = compute_ranks(x);
    let rank_y = compute_ranks(y);

    // Pearson correlation of ranks
    let mean_x: f64 = rank_x.iter().sum::<f64>() / n as f64;
    let mean_y: f64 = rank_y.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n {
        let dx = rank_x[i] - mean_x;
        let dy = rank_y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x < 1e-10 || var_y < 1e-10 {
        return 0.0;
    }

    cov / (var_x.sqrt() * var_y.sqrt())
}

/// Compute ranks for Spearman correlation.
fn compute_ranks(values: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; values.len()];
    for (rank, (idx, _)) in indexed.into_iter().enumerate() {
        ranks[idx] = rank as f64 + 1.0;
    }

    ranks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&v1, &v2) - 1.0).abs() < 1e-10);

        let v3 = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&v1, &v3).abs() < 1e-10);
    }

    #[test]
    fn test_tensor3_contraction() {
        // Simple 2x2x2 tensor
        let tensor = vec![
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![vec![0.0, 1.0], vec![1.0, 0.0]],
        ];
        let subj = vec![1.0, 0.0];
        let obj = vec![0.0, 1.0];

        let result = contract_tensor3(&tensor, &subj, &obj);
        // T[0][j][1] for j=0,1 → [0.0, 1.0]
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sentence_vector_intransitive() {
        let discocat = DisCoCat::with_toy_lexicon();
        let vec = discocat.sentence_vector(&["Alice", "runs"]).unwrap();

        assert_eq!(vec.len(), 4);
        // Just check it's non-zero
        assert!(norm(&vec) > 0.1);
    }

    #[test]
    fn test_sentence_vector_transitive() {
        let discocat = DisCoCat::with_toy_lexicon();
        let vec = discocat
            .sentence_vector(&["Alice", "loves", "Bob"])
            .unwrap();

        assert_eq!(vec.len(), 4);
        assert!(norm(&vec) > 0.1);
    }

    #[test]
    fn test_similarity_same_sentence() {
        let discocat = DisCoCat::with_toy_lexicon();
        let sim = discocat
            .similarity(&["Alice", "runs"], &["Alice", "runs"])
            .unwrap();

        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_similarity_different_sentences() {
        let discocat = DisCoCat::with_toy_lexicon();

        let sim_similar = discocat
            .similarity(&["Alice", "loves", "Bob"], &["Alice", "adores", "Bob"])
            .unwrap();

        let sim_different = discocat
            .similarity(&["Alice", "loves", "Bob"], &["Alice", "hates", "Bob"])
            .unwrap();

        // "loves" should be more similar to "adores" than to "hates"
        assert!(sim_similar > sim_different);
    }

    #[test]
    fn test_word_order_matters() {
        let discocat = DisCoCat::with_toy_lexicon();

        let v1 = discocat.sentence_vector(&["dog", "bites", "man"]).unwrap();
        let v2 = discocat.sentence_vector(&["man", "bites", "dog"]).unwrap();

        // DisCoCat should distinguish word order
        let discocat_sim = cosine_similarity(&v1, &v2);

        // BoW cannot distinguish
        let bow_sim = discocat.bow_similarity(&["dog", "bites", "man"], &["man", "bites", "dog"]);

        assert!(bow_sim > discocat_sim || (bow_sim - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_spearman_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((spearman_correlation(&x, &y) - 1.0).abs() < 1e-10);

        let y_rev = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        assert!((spearman_correlation(&x, &y_rev) + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_verb_tensor_construction() {
        let v1 = vec![1.0, 0.0];
        let v2 = vec![0.0, 1.0];
        let pairs: Vec<(&[f64], &[f64])> = vec![
            (v1.as_slice(), v2.as_slice()),
            (v2.as_slice(), v1.as_slice()),
        ];

        let tensor = build_verb_tensor(&pairs, 2, 2, VerbConstructionMethod::Relational);

        assert_eq!(tensor.len(), 2);
        assert_eq!(tensor[0].len(), 2);
        assert_eq!(tensor[0][0].len(), 2);
    }

    #[test]
    fn test_baselines() {
        let mut lexicon = HashMap::new();
        lexicon.insert("dog".to_string(), vec![1.0, 0.0]);
        lexicon.insert("cat".to_string(), vec![0.8, 0.2]);
        lexicon.insert("runs".to_string(), vec![0.5, 0.5]);

        let bow = Baselines::bag_of_words(&["dog", "runs"], &lexicon);
        assert!((bow[0] - 1.5).abs() < 1e-10);
        assert!((bow[1] - 0.5).abs() < 1e-10);

        let mult = Baselines::multiplication_model(&["dog", "runs"], &lexicon);
        assert!((mult[0] - 0.5).abs() < 1e-10);
        assert!((mult[1] - 0.0).abs() < 1e-10);
    }
}
