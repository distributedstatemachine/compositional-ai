//! # Diff - Autodifferentiation via Category Theory (Sessions 7-10)
//!
//! This crate implements automatic differentiation using categorical semantics,
//! as described in Fong/Spivak/Tuyéras "Backprop as Functor".
//!
//! ## Core Concepts
//!
//! - **Computation graphs are string diagrams** — boxes are ops, wires are tensors
//! - **Forward pass is a functor** — preserves composition structure
//! - **Backward pass is a functor to the opposite category** — reverses arrows
//! - **Const generics** — compile-time dimension checking
//!
//! ## Modules
//!
//! - [`tensor`] — Const generic tensors with compile-time shape checking (Session 7)
//! - `ops` — Differentiable operations (Session 7, coming)
//! - `forward` — Forward evaluation (Session 8, coming)
//! - `backward` — Reverse-mode autodiff (Session 9, coming)
//! - `optim` — Gradient descent optimization (Session 10, coming)
//!
//! ## Example
//!
//! ```rust
//! use compositional_diff::tensor::{Tensor, Linear, MLP};
//!
//! // All dimension checking at compile time!
//! let mlp: MLP<784, 256, 10> = MLP::random();
//! let input: Tensor<f32, 1, 784> = Tensor::zeros();
//! let output: Tensor<f32, 1, 10> = mlp.forward(&input);
//!
//! // Wrong dimensions won't compile:
//! // let bad: Tensor<f32, 1, 100> = Tensor::zeros();
//! // let _ = mlp.forward(&bad);  // Error!
//! ```

pub mod tensor;

// Re-export key types
pub use tensor::{Linear, Tensor, MLP};
