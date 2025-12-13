//! # Error Types
//!
//! Errors in compositional AI are first-class: they represent
//! "non-composable morphisms" — attempts to wire incompatible components.
//!
//! This aligns with the categorical view: composition `f ; g` is only
//! defined when `cod(f) = dom(g)`. A shape mismatch is not a "bug" —
//! it's a mathematically undefined operation.

use thiserror::Error;

use crate::shape::Shape;

/// Core errors for the compositional AI system.
///
/// These represent composition failures — attempts to connect
/// components with incompatible interfaces.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum CoreError {
    /// Shapes don't match at a composition boundary.
    /// This is the categorical equivalent of `cod(f) ≠ dom(g)`.
    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: Shape, got: Shape },

    /// Attempted to connect ports that don't exist.
    #[error("Invalid port index: {index} (node has {count} ports)")]
    InvalidPort { index: usize, count: usize },

    /// Diagram validation failed.
    #[error("Diagram validation failed: {reason}")]
    ValidationError { reason: String },

    /// Category composition undefined (morphisms don't chain).
    #[error("Cannot compose: codomain {codomain} ≠ domain {domain}")]
    CompositionUndefined { codomain: String, domain: String },
}
