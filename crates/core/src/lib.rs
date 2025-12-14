//! # Core - Compositional AI Foundations
//!
//! This crate provides the foundational abstractions for compositional AI:
//!
//! - **Shapes**: Type-level descriptions of tensor dimensions (Session 2)
//! - **Errors**: First-class composition failures (Session 2)
//! - **Categories**: Finite categories and functors (Session 3)
//! - **Diagrams**: String diagram data structures for wiring computations (Session 4+)
//! - **Capabilities**: Yoneda-style extensible scope system (Session 3.6)
//!
//! ## Design Philosophy
//!
//! "Composition-first" means treating pipelines as first-class values.
//! Instead of imperative "run A, then run B", we build composite morphisms
//! that can be inspected, transformed, and verified before execution.

pub mod capability;
pub mod cat;
pub mod diagram;
pub mod error;
pub mod shape;

// Re-export key types at crate root for convenience
pub use capability::{Capability, CapabilityError, CapabilityScope, Handles, Request};
pub use cat::{Coproduct, Scope};
pub use diagram::{Diagram, Node, Port};
pub use error::CoreError;
pub use shape::{Shape, TypeId};
