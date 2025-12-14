//! # Training Utilities (Session 10)
//!
//! This module provides utilities for training models:
//!
//! - Loss functions (MSE, Cross-Entropy)
//! - Training loop helpers
//! - Simple model abstractions
//!
//! ## Example: Training Linear Regression
//!
//! ```rust
//! use compositional_diff::train::{mse_loss_graph, train_step};
//! use compositional_diff::optim::SGD;
//! use compositional_diff::ops::RTensor;
//! use compositional_diff::forward::DiffGraph;
//!
//! // Create simple linear model: y = w * x
//! let mut w = RTensor::randn_seeded(vec![1, 1], 0.1, 42);
//!
//! // Synthetic data: y = 2 * x
//! let x = RTensor::matrix(1, 1, vec![1.0]);
//! let y_true = RTensor::matrix(1, 1, vec![2.0]);
//!
//! let optimizer = SGD::new(0.1);
//!
//! // Training step
//! let (loss, grad_w) = train_step(&x, &y_true, &w);
//! optimizer.step(&mut [w], &[grad_w]);
//! ```

use crate::backward::grad;
use crate::forward::DiffGraph;
use crate::ops::RTensor;

/// Compute MSE loss between prediction and target.
///
/// MSE = mean((pred - target)²)
pub fn mse_loss(prediction: &RTensor, target: &RTensor) -> f32 {
    assert_eq!(prediction.shape, target.shape);
    let diff = prediction.sub(target);
    let squared: f32 = diff.data.iter().map(|x| x * x).sum();
    squared / prediction.data.len() as f32
}

/// Build a graph for MSE loss computation.
///
/// Returns (loss_node, graph) where the graph has inputs:
/// - 0: prediction
/// - 1: target
pub fn mse_loss_graph(shape: Vec<usize>) -> (petgraph::graph::NodeIndex, DiffGraph) {
    let mut graph = DiffGraph::new();
    let pred = graph.input(0, shape.clone());
    let target = graph.input(1, shape);

    // diff = pred - target
    let diff = graph.sub(pred, target);

    // squared = diff * diff
    let squared = graph.mul(diff, diff);

    // loss = sum(squared) -- this gives sum, not mean, but grad is same up to scale
    let loss = graph.sum_all(squared);

    graph.mark_output(loss);
    (loss, graph)
}

/// Perform a single training step for linear regression.
///
/// Model: y = W @ x
///
/// Returns (loss, gradient_for_W)
pub fn train_step(x: &RTensor, y_true: &RTensor, w: &RTensor) -> (f32, RTensor) {
    // Build computation graph: loss = sum((W @ x - y)²)
    let mut graph = DiffGraph::new();

    // Inputs: W, x, y
    let w_node = graph.input(0, w.shape.clone());
    let x_node = graph.input(1, x.shape.clone());
    let y_node = graph.input(2, y_true.shape.clone());

    // Forward: pred = W @ x
    let pred = graph.matmul(w_node, x_node);

    // Loss: (pred - y)²
    let diff = graph.sub(pred, y_node);
    let squared = graph.mul(diff, diff);
    let loss = graph.sum_all(squared);
    graph.mark_output(loss);

    // Forward pass
    let inputs = vec![w.clone(), x.clone(), y_true.clone()];
    let outputs = graph.forward(&inputs);
    let loss_val = outputs[0].as_scalar();

    // Backward pass
    let grads = grad(&graph, &inputs);

    // Return loss and gradient for W (index 0)
    (loss_val, grads[0].clone())
}

/// Train a linear model on data.
///
/// Model: y = W @ x
///
/// Returns the loss history.
pub fn train_linear(
    w: &mut RTensor,
    data: &[(RTensor, RTensor)],
    epochs: usize,
    learning_rate: f32,
) -> Vec<f32> {
    use crate::optim::SGD;

    let optimizer = SGD::new(learning_rate);
    let mut losses = Vec::new();

    for _epoch in 0..epochs {
        let mut epoch_loss = 0.0;

        for (x, y_true) in data {
            let (loss, grad_w) = train_step(x, y_true, w);
            epoch_loss += loss;

            // Update weights
            optimizer.step(std::slice::from_mut(w), std::slice::from_ref(&grad_w));

            // Apply the update to the actual w
            for (wp, gp) in w.data.iter_mut().zip(grad_w.data.iter()) {
                *wp -= learning_rate * gp;
            }
        }

        losses.push(epoch_loss / data.len() as f32);
    }

    losses
}

/// Generate synthetic linear regression data.
///
/// Generates data points (x, y) where y = slope * x + intercept + noise.
pub fn generate_linear_data(
    n_samples: usize,
    slope: f32,
    intercept: f32,
    noise_scale: f32,
    seed: u64,
) -> Vec<(RTensor, RTensor)> {
    let mut data = Vec::with_capacity(n_samples);
    let mut state = seed;

    for _ in 0..n_samples {
        // Generate x uniformly in [-5, 5]
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = (state as f32 / u64::MAX as f32) * 10.0 - 5.0;

        // Generate noise
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u1 = (state as f32 / u64::MAX as f32).max(1e-10);
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u2 = state as f32 / u64::MAX as f32;
        let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * noise_scale;

        // y = slope * x + intercept + noise
        let y = slope * x + intercept + noise;

        data.push((
            RTensor::matrix(1, 1, vec![x]),
            RTensor::matrix(1, 1, vec![y]),
        ));
    }

    data
}

/// A simple linear layer: y = W @ x + b
#[derive(Debug, Clone)]
pub struct LinearLayer {
    /// Weight matrix (output_dim, input_dim)
    pub weights: RTensor,
    /// Bias vector (output_dim,)
    pub bias: RTensor,
}

impl LinearLayer {
    /// Create a new linear layer with random weights.
    pub fn new(input_dim: usize, output_dim: usize, seed: u64) -> Self {
        // Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))
        let scale = (2.0 / (input_dim + output_dim) as f32).sqrt();
        Self {
            weights: RTensor::randn_seeded(vec![output_dim, input_dim], scale, seed),
            bias: RTensor::zeros(vec![output_dim]),
        }
    }

    /// Forward pass: y = W @ x + b
    pub fn forward(&self, x: &RTensor) -> RTensor {
        let wx = self.weights.matmul(x);
        // Add bias (broadcasting - for now assume shapes match)
        if wx.shape == self.bias.shape {
            wx.add(&self.bias)
        } else {
            // Simple case: wx is (out, 1), bias is (out,)
            // Broadcast bias to match
            let bias_col = RTensor::from_data(wx.shape.clone(), self.bias.data.clone());
            wx.add(&bias_col)
        }
    }

    /// Get parameters as a slice.
    pub fn params(&self) -> Vec<&RTensor> {
        vec![&self.weights, &self.bias]
    }

    /// Get mutable parameters.
    pub fn params_mut(&mut self) -> Vec<&mut RTensor> {
        vec![&mut self.weights, &mut self.bias]
    }
}

/// A simple MLP (multi-layer perceptron) with one hidden layer.
#[derive(Debug, Clone)]
pub struct SimpleMLP {
    pub layer1: LinearLayer,
    pub layer2: LinearLayer,
}

impl SimpleMLP {
    /// Create a new MLP with given dimensions.
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, seed: u64) -> Self {
        Self {
            layer1: LinearLayer::new(input_dim, hidden_dim, seed),
            layer2: LinearLayer::new(hidden_dim, output_dim, seed.wrapping_add(1)),
        }
    }

    /// Forward pass: ReLU(W1 @ x + b1) -> W2 @ h + b2
    pub fn forward(&self, x: &RTensor) -> RTensor {
        let h = self.layer1.forward(x).relu();
        self.layer2.forward(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss() {
        let pred = RTensor::vector(vec![1.0, 2.0, 3.0]);
        let target = RTensor::vector(vec![1.0, 2.0, 3.0]);
        let loss = mse_loss(&pred, &target);
        assert!(loss.abs() < 1e-6); // Perfect prediction = 0 loss

        let pred = RTensor::vector(vec![0.0, 0.0, 0.0]);
        let target = RTensor::vector(vec![1.0, 2.0, 3.0]);
        let loss = mse_loss(&pred, &target);
        // MSE = (1 + 4 + 9) / 3 = 14/3 ≈ 4.667
        assert!((loss - 14.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_train_step() {
        // y = 2 * x, start with w = 1
        let w = RTensor::matrix(1, 1, vec![1.0]);
        let x = RTensor::matrix(1, 1, vec![1.0]);
        let y_true = RTensor::matrix(1, 1, vec![2.0]);

        let (loss, grad) = train_step(&x, &y_true, &w);

        // pred = 1 * 1 = 1
        // loss = (1 - 2)² = 1
        assert!((loss - 1.0).abs() < 1e-6);

        // grad_w = 2 * (pred - y) * x = 2 * (-1) * 1 = -2
        assert!((grad.data[0] - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_generate_data() {
        let data = generate_linear_data(10, 2.0, 1.0, 0.0, 42);
        assert_eq!(data.len(), 10);

        // Without noise, y = 2x + 1
        for (x, y) in &data {
            let x_val = x.data[0];
            let y_val = y.data[0];
            let expected = 2.0 * x_val + 1.0;
            assert!((y_val - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_linear_layer() {
        let layer = LinearLayer::new(3, 2, 42);
        assert_eq!(layer.weights.shape, vec![2, 3]);
        assert_eq!(layer.bias.shape, vec![2]);

        let x = RTensor::matrix(3, 1, vec![1.0, 2.0, 3.0]);
        let y = layer.forward(&x);
        assert_eq!(y.shape, vec![2, 1]);
    }

    #[test]
    fn test_training_decreases_loss() {
        // Simple test: train w to learn y = 2x
        let mut w = RTensor::matrix(1, 1, vec![0.0]); // Start at 0
        let data = generate_linear_data(20, 2.0, 0.0, 0.0, 42);

        let losses = train_linear(&mut w, &data, 50, 0.01);

        // Loss should decrease
        assert!(losses.last().unwrap() < losses.first().unwrap());

        // w should be close to 2.0
        assert!((w.data[0] - 2.0).abs() < 0.5);
    }
}
