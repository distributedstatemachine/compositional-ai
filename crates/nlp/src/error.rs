//! Error types for NLP operations.

use thiserror::Error;

/// Errors that can occur in NLP computations.
#[derive(Debug, Clone, Error)]
pub enum NlpError {
    /// Word not found in lexicon.
    #[error("Unknown word: '{word}'")]
    UnknownWord { word: String },

    /// Type mismatch in composition.
    #[error("Type mismatch: expected {expected}, got {got}")]
    TypeMismatch { expected: String, got: String },

    /// Parse error.
    #[error("Parse error: {message}")]
    ParseError { message: String },

    /// Empty sentence.
    #[error("Cannot parse empty sentence")]
    EmptySentence,

    /// Invalid type expression.
    #[error("Invalid type expression: {expression}")]
    InvalidType { expression: String },
}
