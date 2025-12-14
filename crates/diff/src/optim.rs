//! # Optimization (Session 10)
//!
//! This module provides optimizers for gradient-based training.
//!
//! ## Optimizers
//!
//! - [`SGD`]: Stochastic Gradient Descent
//!
//! ## Example
//!
//! ```rust
//! use compositional_diff::optim::SGD;
//! use compositional_diff::ops::RTensor;
//!
//! let mut params = vec![RTensor::randn(vec![3, 2], 0.1)];
//! let grads = vec![RTensor::full(vec![3, 2], 0.5)];
//!
//! let optimizer = SGD::new(0.01);
//! optimizer.step(&mut params, &grads);
//! // params have been updated: param = param - 0.01 * grad
//! ```

use crate::ops::RTensor;

/// Stochastic Gradient Descent optimizer.
///
/// Updates parameters using: `θ = θ - lr * ∇L`
#[derive(Debug, Clone)]
pub struct SGD {
    /// Learning rate (step size)
    pub learning_rate: f32,
}

impl SGD {
    /// Create a new SGD optimizer with the given learning rate.
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }

    /// Perform a single optimization step.
    ///
    /// Updates each parameter in-place: `param = param - lr * grad`
    pub fn step(&self, params: &mut [RTensor], grads: &[RTensor]) {
        assert_eq!(
            params.len(),
            grads.len(),
            "Number of params ({}) must match number of grads ({})",
            params.len(),
            grads.len()
        );

        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            assert_eq!(
                param.shape, grad.shape,
                "Param shape {:?} must match grad shape {:?}",
                param.shape, grad.shape
            );

            for (p, g) in param.data.iter_mut().zip(grad.data.iter()) {
                *p -= self.learning_rate * g;
            }
        }
    }

    /// Perform optimization step with gradient clipping.
    ///
    /// Clips gradient values to [-max_grad, max_grad] before updating.
    pub fn step_with_clip(&self, params: &mut [RTensor], grads: &[RTensor], max_grad: f32) {
        assert_eq!(params.len(), grads.len());

        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            let clipped = grad.clip(max_grad);
            for (p, g) in param.data.iter_mut().zip(clipped.data.iter()) {
                *p -= self.learning_rate * g;
            }
        }
    }
}

/// SGD with momentum.
///
/// Updates parameters using:
/// - `v = momentum * v + grad`
/// - `θ = θ - lr * v`
#[derive(Debug, Clone)]
pub struct SGDMomentum {
    /// Learning rate
    pub learning_rate: f32,
    /// Momentum coefficient (typically 0.9)
    pub momentum: f32,
    /// Velocity buffers (one per parameter)
    velocities: Vec<RTensor>,
}

impl SGDMomentum {
    /// Create a new SGD with momentum optimizer.
    pub fn new(learning_rate: f32, momentum: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            velocities: Vec::new(),
        }
    }

    /// Initialize velocity buffers for the given parameters.
    pub fn init(&mut self, params: &[RTensor]) {
        self.velocities = params.iter().map(|p| p.zeros_like()).collect();
    }

    /// Perform a single optimization step with momentum.
    pub fn step(&mut self, params: &mut [RTensor], grads: &[RTensor]) {
        // Initialize velocities if needed
        if self.velocities.is_empty() {
            self.init(params);
        }

        assert_eq!(params.len(), grads.len());
        assert_eq!(params.len(), self.velocities.len());

        for ((param, grad), velocity) in params
            .iter_mut()
            .zip(grads.iter())
            .zip(self.velocities.iter_mut())
        {
            // v = momentum * v + grad
            for (v, g) in velocity.data.iter_mut().zip(grad.data.iter()) {
                *v = self.momentum * *v + g;
            }

            // θ = θ - lr * v
            for (p, v) in param.data.iter_mut().zip(velocity.data.iter()) {
                *p -= self.learning_rate * v;
            }
        }
    }
}

/// Parameter collection for managing named parameters.
#[derive(Debug, Clone, Default)]
pub struct Parameters {
    names: Vec<String>,
    tensors: Vec<RTensor>,
}

impl Parameters {
    /// Create an empty parameter collection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a parameter with a name.
    pub fn register(&mut self, name: &str, tensor: RTensor) {
        self.names.push(name.to_string());
        self.tensors.push(tensor);
    }

    /// Get a parameter by name.
    pub fn get(&self, name: &str) -> Option<&RTensor> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| &self.tensors[i])
    }

    /// Get a mutable reference to a parameter by name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut RTensor> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| &mut self.tensors[i])
    }

    /// Get all parameters as a slice.
    pub fn as_slice(&self) -> &[RTensor] {
        &self.tensors
    }

    /// Get all parameters as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [RTensor] {
        &mut self.tensors
    }

    /// Number of parameters.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Iterate over (name, tensor) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &RTensor)> {
        self.names
            .iter()
            .zip(self.tensors.iter())
            .map(|(n, t)| (n.as_str(), t))
    }

    /// Total number of scalar parameters.
    pub fn num_parameters(&self) -> usize {
        self.tensors.iter().map(|t| t.data.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_step() {
        let mut params = vec![RTensor::vector(vec![1.0, 2.0, 3.0])];
        let grads = vec![RTensor::vector(vec![0.1, 0.2, 0.3])];

        let optimizer = SGD::new(1.0); // lr = 1.0 for easy testing
        optimizer.step(&mut params, &grads);

        // param = param - lr * grad = [1, 2, 3] - 1.0 * [0.1, 0.2, 0.3] = [0.9, 1.8, 2.7]
        assert!((params[0].data[0] - 0.9).abs() < 1e-6);
        assert!((params[0].data[1] - 1.8).abs() < 1e-6);
        assert!((params[0].data[2] - 2.7).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_with_clip() {
        let mut params = vec![RTensor::vector(vec![1.0, 2.0, 3.0])];
        let grads = vec![RTensor::vector(vec![10.0, -10.0, 0.5])]; // Large gradients

        let optimizer = SGD::new(1.0);
        optimizer.step_with_clip(&mut params, &grads, 1.0); // Clip to [-1, 1]

        // Clipped grads: [1.0, -1.0, 0.5]
        // param = [1, 2, 3] - [1.0, -1.0, 0.5] = [0, 3, 2.5]
        assert!((params[0].data[0] - 0.0).abs() < 1e-6);
        assert!((params[0].data[1] - 3.0).abs() < 1e-6);
        assert!((params[0].data[2] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_momentum() {
        let mut params = vec![RTensor::vector(vec![1.0, 2.0])];
        let grads = vec![RTensor::vector(vec![0.1, 0.1])];

        let mut optimizer = SGDMomentum::new(0.1, 0.9);

        // First step: v = 0 * 0.9 + grad = grad
        // param = param - 0.1 * grad
        optimizer.step(&mut params, &grads);

        // After first step: param ≈ [0.99, 1.99]
        assert!((params[0].data[0] - 0.99).abs() < 1e-6);

        // Second step: v = 0.9 * grad + grad = 1.9 * grad
        // param = param - 0.1 * 1.9 * grad = param - 0.19 * grad
        optimizer.step(&mut params, &grads);

        // After second step: param ≈ [0.99 - 0.019, 1.99 - 0.019] = [0.971, 1.971]
        assert!((params[0].data[0] - 0.971).abs() < 1e-6);
    }

    #[test]
    fn test_parameters_collection() {
        let mut params = Parameters::new();
        params.register("weights", RTensor::matrix(2, 3, vec![1.0; 6]));
        params.register("bias", RTensor::vector(vec![0.0, 0.0]));

        assert_eq!(params.len(), 2);
        assert_eq!(params.num_parameters(), 8); // 6 + 2

        assert!(params.get("weights").is_some());
        assert!(params.get("bias").is_some());
        assert!(params.get("nonexistent").is_none());
    }
}
