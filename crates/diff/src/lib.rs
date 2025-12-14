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
//! - [`ops`] — Differentiable operations (Session 8)
//! - [`forward`] — Forward evaluation via topological sort (Session 8)
//! - [`mod@backward`] — Reverse-mode autodiff (Session 9)
//! - [`optim`] — Gradient descent optimization (Session 10)
//! - [`train`] — Training utilities and loss functions (Session 10)
//!
//! ## Example: Const Generic Tensors (Session 7)
//!
//! ```rust
//! use compositional_diff::tensor::{Tensor, Linear, MLP};
//!
//! // All dimension checking at compile time!
//! let mlp: MLP<784, 256, 10> = MLP::random();
//! let input: Tensor<f32, 1, 784> = Tensor::zeros();
//! let output: Tensor<f32, 1, 10> = mlp.forward(&input);
//! ```
//!
//! ## Example: Forward Evaluation (Session 8)
//!
//! ```rust
//! use compositional_diff::forward::DiffGraph;
//! use compositional_diff::ops::RTensor;
//!
//! // Build: y = ReLU(x + y)
//! let mut graph = DiffGraph::new();
//! let x = graph.input(0, vec![3]);
//! let y = graph.input(1, vec![3]);
//! let sum = graph.add(x, y);
//! let out = graph.relu(sum);
//! graph.mark_output(out);
//!
//! // Evaluate
//! let a = RTensor::vector(vec![-1.0, 2.0, -3.0]);
//! let b = RTensor::vector(vec![2.0, -1.0, 4.0]);
//! let result = graph.forward(&[a, b]);
//! assert_eq!(result[0].data, vec![1.0, 1.0, 1.0]);
//! ```

pub mod backward;
pub mod forward;
pub mod ops;
pub mod optim;
pub mod tensor;
pub mod train;

// Re-export key types
pub use backward::{backward, grad, grad_check, ForwardCache, ForwardValue, GradCheckError};
pub use forward::{DiffGraph, ForwardEval};
pub use ops::{DiffOp, RTensor};
pub use optim::{Parameters, SGDMomentum, SGD};
pub use tensor::{Linear, Tensor, MLP};
pub use train::{mse_loss, train_step, LinearLayer, SimpleMLP};
