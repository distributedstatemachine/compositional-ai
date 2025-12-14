//! Error types for probability operations.

use thiserror::Error;

/// Errors that can occur in probability computations.
#[derive(Debug, Clone, Error)]
pub enum ProbError {
    /// Distribution doesn't sum to 1.
    #[error("Distribution not normalized: sum = {sum} (expected 1.0)")]
    NotNormalized { sum: f32 },

    /// Negative probability encountered.
    #[error("Negative probability encountered")]
    NegativeProbability,

    /// All weights are zero (can't normalize).
    #[error("Cannot normalize: all weights are zero")]
    ZeroWeights,

    /// Empty distribution.
    #[error("Distribution cannot be empty")]
    EmptyDistribution,

    /// Empty kernel.
    #[error("Kernel cannot be empty")]
    EmptyKernel,

    /// Rows have different lengths.
    #[error("Kernel has ragged rows (rows have different lengths)")]
    RaggedMatrix,

    /// A row doesn't sum to 1.
    #[error("Row {row} not normalized: sum = {sum} (expected 1.0)")]
    RowNotNormalized { row: usize, sum: f32 },

    /// Shape mismatch for composition or application.
    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: usize, got: usize },

    /// Index out of bounds.
    #[error("Index {index} out of bounds for size {size}")]
    IndexOutOfBounds { index: usize, size: usize },
}
