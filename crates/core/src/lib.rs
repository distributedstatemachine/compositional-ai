//! # Core - Compositional AI Foundations
//!
//! This crate provides the foundational abstractions for compositional AI:
//!
//! - **Shapes**: Type-level descriptions of tensor dimensions (Session 2)
//! - **Errors**: First-class composition failures (Session 2)
//! - **Categories**: Finite categories and functors (Session 3)
//! - **Diagrams**: String diagram data structures for wiring computations (Session 4+)
//!
//! ## Design Philosophy
//!
//! "Composition-first" means treating pipelines as first-class values.
//! Instead of imperative "run A, then run B", we build composite morphisms
//! that can be inspected, transformed, and verified before execution.

pub mod cat;
pub mod diagram;
pub mod error;
pub mod shape;

// Re-export key types at crate root for convenience
pub use error::CoreError;
pub use shape::{Shape, TypeId};
