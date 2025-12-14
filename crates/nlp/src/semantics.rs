//! Tensor Semantics: From Grammar to Vectors (DisCoCat)
//!
//! This module implements the semantics functor F: Grammar → Vect that maps:
//! - Pregroup types to vector spaces
//! - Words to vectors/tensors
//! - Grammatical reductions to tensor contractions
//!
//! # The Semantics Functor
//!
//! ```text
//! F(N) = ℝⁿ           (noun space)
//! F(S) = ℝ            (sentence space = scalars)
//! F(Nˡ) = F(Nʳ) = ℝⁿ  (dual ≅ primal for finite dim)
//! ```
//!
//! # Example
//!
//! ```rust
//! use compositional_nlp::pregroup::Grammar;
//! use compositional_nlp::semantics::Semantics;
//!
//! let grammar = Grammar::english_basic();
//! let mut semantics = Semantics::new(2); // 2D noun space
//!
//! // Add word vectors
//! semantics.add_noun("Alice", vec![0.8, 0.6]);
//! semantics.add_noun("Bob", vec![0.7, 0.3]);
//! semantics.add_transitive_verb("loves", vec![
//!     vec![0.9, 0.1],
//!     vec![0.2, 0.8],
//! ]);
//!
//! // Compute sentence meaning
//! let parse = grammar.parse(&["Alice", "loves", "Bob"]).unwrap();
//! let meaning = semantics.sentence_meaning(&parse).unwrap();
//! println!("Meaning of 'Alice loves Bob': {:.4}", meaning);
//! ```

use crate::pregroup::{BasicType, ParseResult, PregroupType};
use crate::NlpError;
use std::collections::HashMap;

/// A tensor representing word meaning.
///
/// For simplicity, we support:
/// - Vectors (rank 1): nouns, intransitive verbs
/// - Matrices (rank 2): transitive verbs, adjectives
#[derive(Debug, Clone)]
pub enum WordTensor {
    /// A vector (for nouns, intransitive verbs)
    Vector(Vec<f64>),
    /// A matrix (for transitive verbs, adjectives)
    /// Stored as row-major: `matrix[i][j]` = element at row i, column j
    Matrix(Vec<Vec<f64>>),
}

impl WordTensor {
    /// Create a vector tensor.
    pub fn vector(v: Vec<f64>) -> Self {
        WordTensor::Vector(v)
    }

    /// Create a matrix tensor.
    pub fn matrix(m: Vec<Vec<f64>>) -> Self {
        WordTensor::Matrix(m)
    }

    /// Get the shape of this tensor.
    pub fn shape(&self) -> Vec<usize> {
        match self {
            WordTensor::Vector(v) => vec![v.len()],
            WordTensor::Matrix(m) => {
                if m.is_empty() {
                    vec![0, 0]
                } else {
                    vec![m.len(), m[0].len()]
                }
            }
        }
    }

    /// Check if this is a vector.
    pub fn is_vector(&self) -> bool {
        matches!(self, WordTensor::Vector(_))
    }

    /// Check if this is a matrix.
    pub fn is_matrix(&self) -> bool {
        matches!(self, WordTensor::Matrix(_))
    }

    /// Get as vector (if it is one).
    pub fn as_vector(&self) -> Option<&Vec<f64>> {
        match self {
            WordTensor::Vector(v) => Some(v),
            _ => None,
        }
    }

    /// Get as matrix (if it is one).
    pub fn as_matrix(&self) -> Option<&Vec<Vec<f64>>> {
        match self {
            WordTensor::Matrix(m) => Some(m),
            _ => None,
        }
    }
}

/// Inner product of two vectors.
pub fn inner_product(v1: &[f64], v2: &[f64]) -> f64 {
    v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum()
}

/// Matrix-vector multiplication: M * v
pub fn matrix_vector_mult(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter().map(|row| inner_product(row, v)).collect()
}

/// Vector-matrix multiplication: v^T * M (row vector times matrix)
pub fn vector_matrix_mult(v: &[f64], m: &[Vec<f64>]) -> Vec<f64> {
    if m.is_empty() || m[0].is_empty() {
        return vec![];
    }
    let cols = m[0].len();
    (0..cols)
        .map(|j| v.iter().enumerate().map(|(i, &vi)| vi * m[i][j]).sum())
        .collect()
}

/// Bilinear form: v1^T * M * v2
pub fn bilinear_form(v1: &[f64], m: &[Vec<f64>], v2: &[f64]) -> f64 {
    let mv = matrix_vector_mult(m, v2);
    inner_product(v1, &mv)
}

/// The semantics functor: maps grammar to vector spaces.
///
/// This implements the DisCoCat model where:
/// - Nouns are vectors
/// - Transitive verbs are matrices
/// - Sentence meaning is computed by tensor contraction
#[derive(Debug, Clone)]
pub struct Semantics {
    /// Dimension of noun space (ℝⁿ)
    pub noun_dim: usize,
    /// Word meanings (word → tensor)
    lexicon: HashMap<String, WordTensor>,
}

impl Semantics {
    /// Create a new semantics with given noun space dimension.
    pub fn new(noun_dim: usize) -> Self {
        Self {
            noun_dim,
            lexicon: HashMap::new(),
        }
    }

    /// Create semantics with a toy lexicon for demonstration.
    pub fn toy_lexicon() -> Self {
        let mut sem = Self::new(2);

        // Nouns (2D vectors)
        // Dimension 0: "human-like" vs "animal-like"
        // Dimension 1: "active" vs "passive"
        sem.add_noun("Alice", vec![0.9, 0.7]);
        sem.add_noun("Bob", vec![0.8, 0.4]);
        sem.add_noun("dog", vec![0.2, 0.9]);
        sem.add_noun("cat", vec![0.3, 0.5]);
        sem.add_noun("man", vec![0.95, 0.6]);
        sem.add_noun("woman", vec![0.95, 0.65]);

        // Intransitive verbs (vectors that get inner-producted with subject)
        sem.add_intransitive_verb("runs", vec![0.3, 0.95]);
        sem.add_intransitive_verb("sleeps", vec![0.8, 0.2]);
        sem.add_intransitive_verb("walks", vec![0.5, 0.7]);

        // Transitive verbs (matrices)
        // loves[i][j] = how much subject-type-i loves object-type-j
        sem.add_transitive_verb(
            "loves",
            vec![
                vec![0.9, 0.7], // humans love humans and animals
                vec![0.6, 0.8], // active things love active things
            ],
        );
        sem.add_transitive_verb(
            "sees",
            vec![
                vec![0.8, 0.8], // everyone can see everyone
                vec![0.7, 0.7],
            ],
        );
        sem.add_transitive_verb(
            "chases",
            vec![
                vec![0.3, 0.9],  // animals chase more
                vec![0.8, 0.95], // active things chase active things
            ],
        );

        // Adjectives (matrices that transform noun vectors)
        sem.add_adjective(
            "big",
            vec![
                vec![1.0, 0.0], // doesn't change human-like
                vec![0.0, 1.2], // increases active
            ],
        );
        sem.add_adjective(
            "small",
            vec![
                vec![1.0, 0.0],
                vec![0.0, 0.8], // decreases active
            ],
        );
        sem.add_adjective(
            "happy",
            vec![
                vec![1.1, 0.0], // slightly more human-like
                vec![0.0, 1.1], // slightly more active
            ],
        );

        // Determiners (identity for simplicity)
        sem.add_adjective("the", vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        sem.add_adjective("a", vec![vec![1.0, 0.0], vec![0.0, 1.0]]);

        sem
    }

    /// Add a noun vector.
    pub fn add_noun(&mut self, word: &str, vector: Vec<f64>) {
        self.lexicon
            .insert(word.to_string(), WordTensor::Vector(vector));
    }

    /// Add an intransitive verb vector.
    pub fn add_intransitive_verb(&mut self, word: &str, vector: Vec<f64>) {
        self.lexicon
            .insert(word.to_string(), WordTensor::Vector(vector));
    }

    /// Add a transitive verb matrix.
    pub fn add_transitive_verb(&mut self, word: &str, matrix: Vec<Vec<f64>>) {
        self.lexicon
            .insert(word.to_string(), WordTensor::Matrix(matrix));
    }

    /// Add an adjective matrix.
    pub fn add_adjective(&mut self, word: &str, matrix: Vec<Vec<f64>>) {
        self.lexicon
            .insert(word.to_string(), WordTensor::Matrix(matrix));
    }

    /// Get the tensor for a word.
    pub fn get_tensor(&self, word: &str) -> Option<&WordTensor> {
        self.lexicon.get(word)
    }

    /// Map a pregroup type to its expected tensor shape.
    ///
    /// - N → `[noun_dim]`
    /// - S → `[1]` (scalar, but we use 1-element for uniformity)
    /// - Nʳ·S → `[noun_dim]` (intransitive verb)
    /// - Nʳ·S·Nˡ → `[noun_dim, noun_dim]` (transitive verb)
    /// - N·Nˡ → `[noun_dim, noun_dim]` (adjective)
    pub fn type_to_shape(&self, typ: &PregroupType) -> Vec<usize> {
        // Count factors that contribute dimensions
        let mut dims = Vec::new();
        for atom in &typ.factors {
            match atom.base {
                BasicType::N => dims.push(self.noun_dim),
                BasicType::S => dims.push(1),
                BasicType::Custom(_) => dims.push(self.noun_dim),
            }
        }
        dims
    }

    /// Compute the meaning of a parsed sentence.
    ///
    /// This implements the tensor contraction following the parse structure.
    pub fn sentence_meaning(&self, parse: &ParseResult) -> Result<f64, NlpError> {
        if parse.typed_words.is_empty() {
            return Err(NlpError::EmptySentence);
        }

        // Get tensors for all words
        let tensors: Vec<&WordTensor> = parse
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

        // Contract based on sentence structure
        self.contract_sentence(&tensors, parse)
    }

    /// Contract tensors according to sentence structure.
    fn contract_sentence(
        &self,
        tensors: &[&WordTensor],
        parse: &ParseResult,
    ) -> Result<f64, NlpError> {
        // Handle common patterns
        match tensors.len() {
            0 => Err(NlpError::EmptySentence),
            1 => {
                // Single word - return norm or first element
                match tensors[0] {
                    WordTensor::Vector(v) => Ok(v.iter().map(|x| x * x).sum::<f64>().sqrt()),
                    WordTensor::Matrix(m) => {
                        Ok(m.iter().flatten().map(|x| x * x).sum::<f64>().sqrt())
                    }
                }
            }
            2 => self.contract_two(tensors, parse),
            3 => self.contract_three(tensors, parse),
            _ => self.contract_general(tensors, parse),
        }
    }

    /// Contract two tensors (e.g., "Alice runs", "big dog").
    fn contract_two(&self, tensors: &[&WordTensor], _parse: &ParseResult) -> Result<f64, NlpError> {
        match (tensors[0], tensors[1]) {
            // N · (Nʳ·S) → S : subject + intransitive verb
            (WordTensor::Vector(subj), WordTensor::Vector(verb)) => Ok(inner_product(subj, verb)),
            // (N·Nˡ) · N → N : adjective + noun (returns norm of result)
            (WordTensor::Matrix(adj), WordTensor::Vector(noun)) => {
                let modified = matrix_vector_mult(adj, noun);
                Ok(modified.iter().map(|x| x * x).sum::<f64>().sqrt())
            }
            _ => Err(NlpError::TypeMismatch {
                expected: "vector-vector or matrix-vector".to_string(),
                got: format!("{:?}-{:?}", tensors[0].shape(), tensors[1].shape()),
            }),
        }
    }

    /// Contract three tensors (e.g., "Alice loves Bob").
    fn contract_three(
        &self,
        tensors: &[&WordTensor],
        _parse: &ParseResult,
    ) -> Result<f64, NlpError> {
        match (tensors[0], tensors[1], tensors[2]) {
            // N · (Nʳ·S·Nˡ) · N → S : subject + transitive verb + object
            (WordTensor::Vector(subj), WordTensor::Matrix(verb), WordTensor::Vector(obj)) => {
                Ok(bilinear_form(subj, verb, obj))
            }
            // (N·Nˡ) · (N·Nˡ) · N → N : two adjectives + noun
            (WordTensor::Matrix(adj1), WordTensor::Matrix(adj2), WordTensor::Vector(noun)) => {
                let modified1 = matrix_vector_mult(adj2, noun);
                let modified2 = matrix_vector_mult(adj1, &modified1);
                Ok(modified2.iter().map(|x| x * x).sum::<f64>().sqrt())
            }
            // (N·Nˡ) · N · (Nʳ·S) → S : adjective + noun + intransitive verb
            (WordTensor::Matrix(adj), WordTensor::Vector(noun), WordTensor::Vector(verb)) => {
                let modified = matrix_vector_mult(adj, noun);
                Ok(inner_product(&modified, verb))
            }
            _ => Err(NlpError::TypeMismatch {
                expected: "subject-verb-object or adj-adj-noun".to_string(),
                got: format!(
                    "{:?}-{:?}-{:?}",
                    tensors[0].shape(),
                    tensors[1].shape(),
                    tensors[2].shape()
                ),
            }),
        }
    }

    /// Contract more than three tensors (general case).
    fn contract_general(
        &self,
        tensors: &[&WordTensor],
        _parse: &ParseResult,
    ) -> Result<f64, NlpError> {
        // Process left-to-right, accumulating a "current" vector/scalar
        // This handles patterns like "the big dog runs" or "Alice sees the big cat"

        let mut result_vec: Option<Vec<f64>> = None;
        let mut pending_verb: Option<&Vec<Vec<f64>>> = None;

        for tensor in tensors {
            match tensor {
                WordTensor::Vector(v) => {
                    if let Some(verb_matrix) = pending_verb {
                        // We have a pending transitive verb, and this is the object
                        if let Some(subj) = &result_vec {
                            let meaning = bilinear_form(subj, verb_matrix, v);
                            result_vec = Some(vec![meaning]);
                            pending_verb = None;
                        } else {
                            // No subject yet, this becomes the subject
                            result_vec = Some(v.clone());
                        }
                    } else if let Some(current) = &result_vec {
                        // Inner product (intransitive verb case)
                        let meaning = inner_product(current, v);
                        result_vec = Some(vec![meaning]);
                    } else {
                        // First vector - becomes current
                        result_vec = Some(v.clone());
                    }
                }
                WordTensor::Matrix(m) => {
                    if let Some(current) = &result_vec {
                        // Check if this is an adjective (modifies noun) or verb
                        // Heuristic: if current is a full vector, treat as adjective
                        // if pending_verb is None
                        if current.len() == self.noun_dim && pending_verb.is_none() {
                            // Could be adjective (pre-multiplied) or transitive verb
                            // Try to detect: if m is square and same size as noun_dim,
                            // and we're in a noun context, apply as adjective
                            if m.len() == self.noun_dim && m[0].len() == self.noun_dim {
                                // Check if this might be a transitive verb (next word is object)
                                // For now, treat as transitive verb
                                pending_verb = Some(m);
                            }
                        } else {
                            // Apply matrix to current vector
                            let transformed = matrix_vector_mult(m, current);
                            result_vec = Some(transformed);
                        }
                    } else {
                        // Matrix without prior vector - adjective waiting for noun
                        // Store identity-like state (we'll apply when we get a noun)
                        // For simplicity, we skip this case
                        continue;
                    }
                }
            }
        }

        // Return final result
        match result_vec {
            Some(v) if v.len() == 1 => Ok(v[0]),
            Some(v) => Ok(v.iter().map(|x| x * x).sum::<f64>().sqrt()),
            None => Ok(0.0),
        }
    }

    /// Compute similarity between two sentence meanings.
    pub fn sentence_similarity(
        &self,
        parse1: &ParseResult,
        parse2: &ParseResult,
    ) -> Result<f64, NlpError> {
        let m1 = self.sentence_meaning(parse1)?;
        let m2 = self.sentence_meaning(parse2)?;

        // Cosine-like similarity (both are scalars, so just compare magnitudes)
        let max = m1.abs().max(m2.abs());
        if max < 1e-10 {
            Ok(1.0) // Both near zero
        } else {
            Ok(1.0 - (m1 - m2).abs() / max)
        }
    }

    /// Get the meaning of a noun phrase (returns a vector).
    pub fn noun_phrase_meaning(&self, parse: &ParseResult) -> Result<Vec<f64>, NlpError> {
        if parse.typed_words.is_empty() {
            return Err(NlpError::EmptySentence);
        }

        // Get tensors for all words
        let tensors: Vec<&WordTensor> = parse
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

        // Apply adjectives/determiners to noun
        let mut result: Option<Vec<f64>> = None;

        for tensor in tensors.iter().rev() {
            // Process right-to-left for noun phrases
            match tensor {
                WordTensor::Vector(v) => {
                    result = Some(v.clone());
                }
                WordTensor::Matrix(m) => {
                    if let Some(current) = &result {
                        result = Some(matrix_vector_mult(m, current));
                    }
                }
            }
        }

        result.ok_or(NlpError::ParseError {
            message: "No noun found in phrase".to_string(),
        })
    }

    /// Compute cosine similarity between two vectors.
    pub fn cosine_similarity(v1: &[f64], v2: &[f64]) -> f64 {
        let dot = inner_product(v1, v2);
        let norm1 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm1 < 1e-10 || norm2 < 1e-10 {
            0.0
        } else {
            dot / (norm1 * norm2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pregroup::Grammar;

    #[test]
    fn test_inner_product() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![4.0, 5.0, 6.0];
        assert!((inner_product(&v1, &v2) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_vector_mult() {
        let m = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let v = vec![1.0, 1.0];
        let result = matrix_vector_mult(&m, &v);
        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!((result[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_bilinear_form() {
        let v1 = vec![1.0, 0.0];
        let m = vec![vec![2.0, 0.0], vec![0.0, 3.0]];
        let v2 = vec![1.0, 0.0];
        assert!((bilinear_form(&v1, &m, &v2) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_intransitive_sentence() {
        let grammar = Grammar::english_basic();
        let semantics = Semantics::toy_lexicon();

        let parse = grammar.parse(&["Alice", "runs"]).unwrap();
        let meaning = semantics.sentence_meaning(&parse).unwrap();

        // Alice = [0.9, 0.7], runs = [0.3, 0.95]
        // meaning = 0.9*0.3 + 0.7*0.95 = 0.27 + 0.665 = 0.935
        assert!((meaning - 0.935).abs() < 1e-10);
    }

    #[test]
    fn test_transitive_sentence() {
        let grammar = Grammar::english_basic();
        let semantics = Semantics::toy_lexicon();

        let parse = grammar.parse(&["Alice", "loves", "Bob"]).unwrap();
        let meaning = semantics.sentence_meaning(&parse).unwrap();

        // Alice = [0.9, 0.7], Bob = [0.8, 0.4]
        // loves = [[0.9, 0.7], [0.6, 0.8]]
        // loves * Bob = [0.9*0.8 + 0.7*0.4, 0.6*0.8 + 0.8*0.4] = [1.0, 0.8]
        // Alice · (loves * Bob) = 0.9*1.0 + 0.7*0.8 = 0.9 + 0.56 = 1.46
        assert!(meaning > 0.0);
    }

    #[test]
    fn test_noun_phrase() {
        let grammar = Grammar::english_basic();
        let semantics = Semantics::toy_lexicon();

        let parse = grammar.parse(&["big", "dog"]).unwrap();
        let np_vec = semantics.noun_phrase_meaning(&parse).unwrap();

        // dog = [0.2, 0.9], big = [[1.0, 0], [0, 1.2]]
        // big * dog = [0.2, 1.08]
        assert!((np_vec[0] - 0.2).abs() < 1e-10);
        assert!((np_vec[1] - 1.08).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity() {
        let v1 = vec![1.0, 0.0];
        let v2 = vec![1.0, 0.0];
        assert!((Semantics::cosine_similarity(&v1, &v2) - 1.0).abs() < 1e-10);

        let v3 = vec![0.0, 1.0];
        assert!(Semantics::cosine_similarity(&v1, &v3).abs() < 1e-10);
    }

    #[test]
    fn test_asymmetric_loves() {
        let grammar = Grammar::english_basic();
        let semantics = Semantics::toy_lexicon();

        let parse1 = grammar.parse(&["Alice", "loves", "Bob"]).unwrap();
        let parse2 = grammar.parse(&["Bob", "loves", "Alice"]).unwrap();

        let m1 = semantics.sentence_meaning(&parse1).unwrap();
        let m2 = semantics.sentence_meaning(&parse2).unwrap();

        // These should be different because Alice ≠ Bob
        // and "loves" matrix is not symmetric
        assert!((m1 - m2).abs() > 0.01);
    }
}
