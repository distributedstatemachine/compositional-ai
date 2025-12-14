//! # Differentiable Operations (Session 7-8)
//!
//! This module defines `DiffOp`, the set of operations used in differentiable
//! computation graphs. Each operation knows how to:
//!
//! - Execute forward (compute outputs from inputs)
//! - (Session 9) Compute VJP (vector-Jacobian product for backprop)
//!
//! ## Operations
//!
//! | Op | Forward | Backward (Session 9) |
//! |----|---------|----------------------|
//! | Add | a + b | grad flows to both |
//! | MatMul | A @ B | ∂L/∂A = grad @ Bᵀ |
//! | ReLU | max(0, x) | grad × (x > 0) |
//! | SumAll | sum(x) | broadcast grad |
//! | Copy | x → (x, x, ...) | sum of grads |

use std::fmt;

/// A runtime tensor for graph evaluation.
///
/// Unlike the const-generic `Tensor<T, M, N>` from Session 7, this tensor
/// has dynamic dimensions for flexibility in graph construction.
#[derive(Clone)]
pub struct RTensor {
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Flattened data in row-major order
    pub data: Vec<f32>,
}

impl RTensor {
    /// Create a tensor filled with zeros.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; size],
        }
    }

    /// Create a tensor filled with a constant value.
    pub fn full(shape: Vec<usize>, value: f32) -> Self {
        let size: usize = shape.iter().product();
        Self {
            shape,
            data: vec![value; size],
        }
    }

    /// Create a tensor from data with given shape.
    pub fn from_data(shape: Vec<usize>, data: Vec<f32>) -> Self {
        let expected_size: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_size,
            "Data length {} doesn't match shape {:?}",
            data.len(),
            shape
        );
        Self { shape, data }
    }

    /// Create a scalar tensor.
    pub fn scalar(value: f32) -> Self {
        Self {
            shape: vec![],
            data: vec![value],
        }
    }

    /// Create a 1D vector tensor.
    pub fn vector(data: Vec<f32>) -> Self {
        let len = data.len();
        Self {
            shape: vec![len],
            data,
        }
    }

    /// Create a 2D matrix tensor.
    pub fn matrix(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self {
            shape: vec![rows, cols],
            data,
        }
    }

    /// Check if this is a scalar.
    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty() || (self.shape.len() == 1 && self.shape[0] == 1)
    }

    /// Get the scalar value (panics if not a scalar).
    pub fn as_scalar(&self) -> f32 {
        assert!(self.is_scalar() || self.data.len() == 1);
        self.data[0]
    }

    /// Total number of elements.
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Element-wise addition.
    pub fn add(&self, other: &RTensor) -> RTensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch for add");
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        RTensor {
            shape: self.shape.clone(),
            data,
        }
    }

    /// Element-wise ReLU.
    pub fn relu(&self) -> RTensor {
        let data: Vec<f32> = self.data.iter().map(|x| x.max(0.0)).collect();
        RTensor {
            shape: self.shape.clone(),
            data,
        }
    }

    /// Sum all elements to a scalar.
    pub fn sum_all(&self) -> RTensor {
        let sum: f32 = self.data.iter().sum();
        RTensor::scalar(sum)
    }

    /// Matrix multiplication (2D tensors only).
    pub fn matmul(&self, other: &RTensor) -> RTensor {
        assert_eq!(self.shape.len(), 2, "matmul requires 2D tensor");
        assert_eq!(other.shape.len(), 2, "matmul requires 2D tensor");

        let m = self.shape[0];
        let k = self.shape[1];
        let k2 = other.shape[0];
        let n = other.shape[1];

        assert_eq!(k, k2, "Inner dimensions must match: {} vs {}", k, k2);

        let mut result = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for kk in 0..k {
                    sum += self.data[i * k + kk] * other.data[kk * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        RTensor {
            shape: vec![m, n],
            data: result,
        }
    }

    /// Element-wise multiplication.
    pub fn mul(&self, other: &RTensor) -> RTensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch for mul");
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        RTensor {
            shape: self.shape.clone(),
            data,
        }
    }

    /// Scalar multiplication.
    pub fn scale(&self, scalar: f32) -> RTensor {
        let data: Vec<f32> = self.data.iter().map(|x| x * scalar).collect();
        RTensor {
            shape: self.shape.clone(),
            data,
        }
    }

    /// Apply a function to each element.
    pub fn map(&self, f: impl Fn(f32) -> f32) -> RTensor {
        let data: Vec<f32> = self.data.iter().map(|&x| f(x)).collect();
        RTensor {
            shape: self.shape.clone(),
            data,
        }
    }

    /// Transpose a 2D matrix.
    pub fn transpose(&self) -> RTensor {
        assert_eq!(self.shape.len(), 2, "transpose requires 2D tensor");
        let rows = self.shape[0];
        let cols = self.shape[1];

        let mut result = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                result[j * rows + i] = self.data[i * cols + j];
            }
        }

        RTensor {
            shape: vec![cols, rows],
            data: result,
        }
    }

    /// Broadcast a scalar to match a given shape.
    pub fn broadcast_to(&self, shape: Vec<usize>) -> RTensor {
        if self.shape == shape {
            return self.clone();
        }

        // Handle scalar broadcast
        if self.is_scalar() || self.data.len() == 1 {
            let size: usize = shape.iter().product();
            return RTensor {
                shape,
                data: vec![self.data[0]; size],
            };
        }

        panic!("Cannot broadcast shape {:?} to {:?}", self.shape, shape);
    }

    /// Create a tensor of ones with the same shape.
    pub fn ones_like(&self) -> RTensor {
        RTensor::full(self.shape.clone(), 1.0)
    }

    /// Create a tensor of zeros with the same shape.
    pub fn zeros_like(&self) -> RTensor {
        RTensor::zeros(self.shape.clone())
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &RTensor) -> RTensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch for sub");
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        RTensor {
            shape: self.shape.clone(),
            data,
        }
    }

    /// Negate all elements.
    pub fn neg(&self) -> RTensor {
        let data: Vec<f32> = self.data.iter().map(|x| -x).collect();
        RTensor {
            shape: self.shape.clone(),
            data,
        }
    }

    /// Create a tensor with random values from N(0, scale²).
    ///
    /// Uses a simple linear congruential generator for reproducibility.
    pub fn randn(shape: Vec<usize>, scale: f32) -> RTensor {
        let size: usize = shape.iter().product();
        let mut data = Vec::with_capacity(size);

        // Simple LCG for reproducible "random" numbers
        // For real applications, use rand crate
        let mut seed: u64 = 12345;
        for _ in 0..size {
            // Box-Muller transform for normal distribution
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = (seed as f32) / (u64::MAX as f32);
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = (seed as f32) / (u64::MAX as f32);

            // Avoid log(0)
            let u1 = u1.max(1e-10);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            data.push(z * scale);
        }

        RTensor { shape, data }
    }

    /// Create a tensor with random values, using a seed.
    pub fn randn_seeded(shape: Vec<usize>, scale: f32, seed: u64) -> RTensor {
        let size: usize = shape.iter().product();
        let mut data = Vec::with_capacity(size);

        let mut state = seed;
        for _ in 0..size {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = (state as f32) / (u64::MAX as f32);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = (state as f32) / (u64::MAX as f32);

            let u1 = u1.max(1e-10);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            data.push(z * scale);
        }

        RTensor { shape, data }
    }

    /// Clip values to [-max_val, max_val].
    pub fn clip(&self, max_val: f32) -> RTensor {
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| x.max(-max_val).min(max_val))
            .collect();
        RTensor {
            shape: self.shape.clone(),
            data,
        }
    }

    /// Compute mean of all elements.
    pub fn mean(&self) -> f32 {
        self.data.iter().sum::<f32>() / self.data.len() as f32
    }

    /// Compute L2 norm.
    pub fn norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

impl fmt::Debug for RTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_scalar() {
            write!(f, "RTensor(scalar={})", self.data[0])
        } else if self.shape.len() == 1 {
            write!(f, "RTensor(vec[{}]={:?})", self.shape[0], self.data)
        } else {
            write!(f, "RTensor(shape={:?}, data={:?})", self.shape, self.data)
        }
    }
}

/// Differentiable operations for computation graphs.
///
/// Each variant represents an operation that can be:
/// - Evaluated in the forward pass (this session)
/// - Differentiated in the backward pass (Session 9)
#[derive(Debug, Clone, PartialEq)]
pub enum DiffOp {
    /// Input placeholder - passes through the boundary input
    Input { index: usize },

    /// Constant value
    Const { value: f32 },

    /// Element-wise addition: a + b
    Add,

    /// Element-wise subtraction: a - b
    Sub,

    /// Element-wise multiplication: a * b
    Mul,

    /// Matrix multiplication: A @ B
    MatMul,

    /// Element-wise ReLU: max(0, x)
    ReLU,

    /// Sum all elements to a scalar
    SumAll,

    /// Copy input to multiple outputs (fan-out)
    Copy { n_outputs: usize },

    /// Parameter (learnable weight) with a name
    Param { name: String },
}

impl DiffOp {
    /// Execute the forward pass for this operation.
    ///
    /// Given the input tensors, compute the output tensor(s).
    pub fn forward(&self, inputs: &[RTensor]) -> Vec<RTensor> {
        match self {
            DiffOp::Input { .. } => {
                // Input nodes receive their value from the boundary
                vec![inputs[0].clone()]
            }

            DiffOp::Const { value } => {
                vec![RTensor::scalar(*value)]
            }

            DiffOp::Add => {
                assert_eq!(inputs.len(), 2, "Add requires 2 inputs");
                vec![inputs[0].add(&inputs[1])]
            }

            DiffOp::Sub => {
                assert_eq!(inputs.len(), 2, "Sub requires 2 inputs");
                vec![inputs[0].sub(&inputs[1])]
            }

            DiffOp::Mul => {
                assert_eq!(inputs.len(), 2, "Mul requires 2 inputs");
                vec![inputs[0].mul(&inputs[1])]
            }

            DiffOp::MatMul => {
                assert_eq!(inputs.len(), 2, "MatMul requires 2 inputs");
                vec![inputs[0].matmul(&inputs[1])]
            }

            DiffOp::ReLU => {
                assert_eq!(inputs.len(), 1, "ReLU requires 1 input");
                vec![inputs[0].relu()]
            }

            DiffOp::SumAll => {
                assert_eq!(inputs.len(), 1, "SumAll requires 1 input");
                vec![inputs[0].sum_all()]
            }

            DiffOp::Copy { n_outputs } => {
                assert_eq!(inputs.len(), 1, "Copy requires 1 input");
                vec![inputs[0].clone(); *n_outputs]
            }

            DiffOp::Param { .. } => {
                // Parameters are provided externally, passed through
                vec![inputs[0].clone()]
            }
        }
    }

    /// Number of inputs this operation expects.
    pub fn num_inputs(&self) -> usize {
        match self {
            DiffOp::Input { .. } => 1,
            DiffOp::Const { .. } => 0,
            DiffOp::Add => 2,
            DiffOp::Sub => 2,
            DiffOp::Mul => 2,
            DiffOp::MatMul => 2,
            DiffOp::ReLU => 1,
            DiffOp::SumAll => 1,
            DiffOp::Copy { .. } => 1,
            DiffOp::Param { .. } => 1,
        }
    }

    /// Number of outputs this operation produces.
    pub fn num_outputs(&self) -> usize {
        match self {
            DiffOp::Copy { n_outputs } => *n_outputs,
            _ => 1,
        }
    }

    /// Compute the Vector-Jacobian Product (VJP) for reverse-mode autodiff.
    ///
    /// Given the forward inputs and the gradient of the loss with respect to
    /// the outputs, compute the gradient with respect to the inputs.
    ///
    /// # Arguments
    ///
    /// * `inputs` - The inputs that were passed to this operation during forward
    /// * `output_grads` - The gradient of the loss w.r.t. this operation's outputs
    ///
    /// # Returns
    ///
    /// A vector of gradients, one for each input to this operation.
    pub fn vjp(&self, inputs: &[RTensor], output_grads: &[RTensor]) -> Vec<RTensor> {
        match self {
            DiffOp::Input { .. } => {
                // Input nodes: gradient passes through
                vec![output_grads[0].clone()]
            }

            DiffOp::Const { .. } => {
                // Constants have no inputs to propagate to
                vec![]
            }

            DiffOp::Add => {
                // z = x + y
                // ∂L/∂x = ∂L/∂z, ∂L/∂y = ∂L/∂z
                let grad = &output_grads[0];
                vec![grad.clone(), grad.clone()]
            }

            DiffOp::Sub => {
                // z = x - y
                // ∂L/∂x = ∂L/∂z, ∂L/∂y = -∂L/∂z
                let grad = &output_grads[0];
                vec![grad.clone(), grad.neg()]
            }

            DiffOp::Mul => {
                // z = x * y
                // ∂L/∂x = ∂L/∂z * y, ∂L/∂y = ∂L/∂z * x
                let (x, y) = (&inputs[0], &inputs[1]);
                let grad = &output_grads[0];
                vec![grad.mul(y), grad.mul(x)]
            }

            DiffOp::MatMul => {
                // C = A @ B where A is (m,k), B is (k,n), C is (m,n)
                // ∂L/∂A = (∂L/∂C) @ Bᵀ  -> (m,n) @ (n,k) = (m,k)
                // ∂L/∂B = Aᵀ @ (∂L/∂C)  -> (k,m) @ (m,n) = (k,n)
                let (a, b) = (&inputs[0], &inputs[1]);
                let grad_c = &output_grads[0];

                let grad_a = grad_c.matmul(&b.transpose());
                let grad_b = a.transpose().matmul(grad_c);

                vec![grad_a, grad_b]
            }

            DiffOp::ReLU => {
                // y = max(0, x)
                // ∂L/∂x = ∂L/∂y * (x > 0 ? 1 : 0)
                let x = &inputs[0];
                let grad = &output_grads[0];

                let mask = x.map(|v| if v > 0.0 { 1.0 } else { 0.0 });
                vec![grad.mul(&mask)]
            }

            DiffOp::SumAll => {
                // s = sum(x)
                // ∂L/∂xᵢ = ∂L/∂s for all i (broadcast)
                let x = &inputs[0];
                let grad = &output_grads[0];

                vec![grad.broadcast_to(x.shape.clone())]
            }

            DiffOp::Copy { n_outputs } => {
                // (y₁, y₂, ...) = (x, x, ...)
                // ∂L/∂x = ∂L/∂y₁ + ∂L/∂y₂ + ...
                let mut sum = output_grads[0].clone();
                for g in output_grads.iter().take(*n_outputs).skip(1) {
                    sum = sum.add(g);
                }
                vec![sum]
            }

            DiffOp::Param { .. } => {
                // Parameters: gradient passes through
                vec![output_grads[0].clone()]
            }
        }
    }
}

impl fmt::Display for DiffOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiffOp::Input { index } => write!(f, "Input[{}]", index),
            DiffOp::Const { value } => write!(f, "Const({})", value),
            DiffOp::Add => write!(f, "Add"),
            DiffOp::Sub => write!(f, "Sub"),
            DiffOp::Mul => write!(f, "Mul"),
            DiffOp::MatMul => write!(f, "MatMul"),
            DiffOp::ReLU => write!(f, "ReLU"),
            DiffOp::SumAll => write!(f, "SumAll"),
            DiffOp::Copy { n_outputs } => write!(f, "Copy({})", n_outputs),
            DiffOp::Param { name } => write!(f, "Param({})", name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rtensor_scalar() {
        let t = RTensor::scalar(42.0);
        assert!(t.is_scalar());
        assert_eq!(t.as_scalar(), 42.0);
    }

    #[test]
    fn test_rtensor_vector() {
        let t = RTensor::vector(vec![1.0, 2.0, 3.0]);
        assert_eq!(t.shape, vec![3]);
        assert_eq!(t.size(), 3);
    }

    #[test]
    fn test_rtensor_add() {
        let a = RTensor::vector(vec![1.0, 2.0, 3.0]);
        let b = RTensor::vector(vec![4.0, 5.0, 6.0]);
        let c = a.add(&b);
        assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_rtensor_relu() {
        let a = RTensor::vector(vec![-1.0, 0.0, 1.0, 2.0]);
        let b = a.relu();
        assert_eq!(b.data, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_rtensor_sum_all() {
        let a = RTensor::vector(vec![1.0, 2.0, 3.0, 4.0]);
        let s = a.sum_all();
        assert!(s.is_scalar());
        assert_eq!(s.as_scalar(), 10.0);
    }

    #[test]
    fn test_rtensor_matmul() {
        // [1 2] × [5 6]   [19 22]
        // [3 4]   [7 8] = [43 50]
        let a = RTensor::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = RTensor::matrix(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = a.matmul(&b);

        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_diffop_add() {
        let a = RTensor::vector(vec![1.0, 2.0]);
        let b = RTensor::vector(vec![3.0, 4.0]);
        let result = DiffOp::Add.forward(&[a, b]);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].data, vec![4.0, 6.0]);
    }

    #[test]
    fn test_diffop_relu() {
        let a = RTensor::vector(vec![-1.0, 2.0, -3.0, 4.0]);
        let result = DiffOp::ReLU.forward(&[a]);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].data, vec![0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_diffop_copy() {
        let a = RTensor::scalar(5.0);
        let result = DiffOp::Copy { n_outputs: 3 }.forward(&[a]);

        assert_eq!(result.len(), 3);
        for t in &result {
            assert_eq!(t.as_scalar(), 5.0);
        }
    }

    #[test]
    fn test_diffop_matmul() {
        let a = RTensor::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = RTensor::matrix(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let result = DiffOp::MatMul.forward(&[a, b]);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].shape, vec![2, 2]);
        // [1 2 3] × [7  8 ]   [58  64 ]
        // [4 5 6]   [9  10] = [139 154]
        //           [11 12]
        assert_eq!(result[0].data, vec![58.0, 64.0, 139.0, 154.0]);
    }
}
