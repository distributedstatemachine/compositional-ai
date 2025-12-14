//! # Typed Tensors - Compile-Time Dimension Checking (Session 7)
//!
//! This module provides tensors with const generic dimensions, enabling
//! compile-time verification of tensor operations. Shape mismatches become
//! type errors, caught before the program runs.
//!
//! ## Core Insight
//!
//! A computation graph **is** a string diagram. The boxes are operations,
//! the wires are tensors, and evaluation is a categorical fold over the DAG.
//!
//! With const generics, the type system enforces that wires connect properly:
//!
//! ```text
//!         ┌─────┐
//!    x ───│  W  │───┐
//!         └─────┘   │   ┌─────┐    ┌──────┐
//!                   ├───│  +  │────│ ReLU │──── y
//!         ┌─────┐   │   └─────┘    └──────┘
//!    1 ───│  b  │───┘
//!         └─────┘
//!
//!    y = ReLU(W·x + b)
//! ```
//!
//! ## Example
//!
//! ```rust
//! use compositional_diff::tensor::{Tensor, Linear, MLP};
//!
//! // All dimension checking happens at compile time!
//! let input: Tensor<f32, 1, 784> = Tensor::zeros();
//! let mlp: MLP<784, 256, 10> = MLP::random();
//!
//! // This compiles - dimensions match
//! let output: Tensor<f32, 1, 10> = mlp.forward(&input);
//!
//! // This would NOT compile:
//! // let bad_input: Tensor<f32, 1, 100> = Tensor::zeros();
//! // let _ = mlp.forward(&bad_input);  // Error: expected 784, got 100
//! ```
//!
//! ## Categorical Interpretation
//!
//! - **Objects** = Types like `Tensor<f32, M, N>`
//! - **Morphisms** = Functions between tensor types
//! - **Composition** = Function composition (type-checked by Rust)
//! - **Tensor Product** = Tuples of tensors

use std::fmt;

/// A tensor with compile-time known dimensions.
///
/// The dimensions M (rows) and N (columns) are const generics,
/// meaning they're part of the type itself. This enables:
///
/// - Compile-time shape checking
/// - Zero runtime overhead for dimension verification
/// - IDE autocomplete showing exact dimensions
/// - Monomorphization for optimal performance
///
/// # Type Parameters
///
/// - `T`: Element type (usually `f32` or `f64`)
/// - `M`: Number of rows (const generic)
/// - `N`: Number of columns (const generic)
#[derive(Clone)]
pub struct Tensor<T, const M: usize, const N: usize> {
    /// Row-major data storage
    data: [[T; N]; M],
}

impl<T: Default + Copy, const M: usize, const N: usize> Default for Tensor<T, M, N> {
    fn default() -> Self {
        Self::zeros()
    }
}

impl<T: Default + Copy, const M: usize, const N: usize> Tensor<T, M, N> {
    /// Create a tensor filled with zeros (default values).
    ///
    /// # Example
    ///
    /// ```rust
    /// use compositional_diff::tensor::Tensor;
    ///
    /// let t: Tensor<f32, 3, 4> = Tensor::zeros();
    /// assert_eq!(t.get(0, 0), 0.0);
    /// ```
    pub fn zeros() -> Self {
        Self {
            data: [[T::default(); N]; M],
        }
    }
}

impl<T: Copy, const M: usize, const N: usize> Tensor<T, M, N> {
    /// Create a tensor from raw data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use compositional_diff::tensor::Tensor;
    ///
    /// let t: Tensor<i32, 2, 3> = Tensor::from_data([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    /// ]);
    /// assert_eq!(t.get(1, 2), 6);
    /// ```
    pub fn from_data(data: [[T; N]; M]) -> Self {
        Self { data }
    }

    /// Get an element at position (i, j).
    pub fn get(&self, i: usize, j: usize) -> T {
        self.data[i][j]
    }

    /// Set an element at position (i, j).
    pub fn set(&mut self, i: usize, j: usize, value: T) {
        self.data[i][j] = value;
    }

    /// Get the raw data.
    pub fn data(&self) -> &[[T; N]; M] {
        &self.data
    }
}

impl<T, const M: usize, const N: usize> Tensor<T, M, N> {
    /// Number of rows (compile-time constant).
    pub const ROWS: usize = M;

    /// Number of columns (compile-time constant).
    pub const COLS: usize = N;

    /// Total number of elements.
    pub const SIZE: usize = M * N;
}

// ============================================================================
// Matrix Operations
// ============================================================================

impl<T, const M: usize, const N: usize> Tensor<T, M, N>
where
    T: Default + Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    /// Matrix multiplication with compile-time dimension checking.
    ///
    /// Computes `self @ other` where:
    /// - `self` is M × N
    /// - `other` is N × P
    /// - result is M × P
    ///
    /// **The inner dimension N must match** - this is enforced by the type system!
    ///
    /// # Example
    ///
    /// ```rust
    /// use compositional_diff::tensor::Tensor;
    ///
    /// let a: Tensor<f32, 2, 3> = Tensor::zeros();  // 2×3
    /// let b: Tensor<f32, 3, 4> = Tensor::zeros();  // 3×4
    /// let c: Tensor<f32, 2, 4> = a.matmul(&b);     // 2×4
    ///
    /// // This won't compile - dimensions don't match:
    /// // let bad: Tensor<f32, 5, 4> = Tensor::zeros();
    /// // let _ = a.matmul(&bad);  // Error: expected Tensor<_, 3, 4>
    /// ```
    pub fn matmul<const P: usize>(&self, other: &Tensor<T, N, P>) -> Tensor<T, M, P> {
        let mut result = Tensor::zeros();
        for i in 0..M {
            for j in 0..P {
                let mut sum = T::default();
                for k in 0..N {
                    sum = sum + self.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        result
    }
}

impl<T, const M: usize, const N: usize> Tensor<T, M, N>
where
    T: Default + Copy + std::ops::Add<Output = T>,
{
    /// Element-wise addition (same shape required by type system).
    ///
    /// # Example
    ///
    /// ```rust
    /// use compositional_diff::tensor::Tensor;
    ///
    /// let a: Tensor<f32, 2, 3> = Tensor::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// let b: Tensor<f32, 2, 3> = Tensor::from_data([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);
    /// let c = a.add(&b);
    /// assert_eq!(c.get(0, 0), 2.0);
    /// ```
    pub fn add(&self, other: &Tensor<T, M, N>) -> Tensor<T, M, N> {
        let mut result = Tensor::zeros();
        for i in 0..M {
            for j in 0..N {
                result.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        result
    }

    /// Add a bias vector (broadcasts across rows).
    ///
    /// (M × N) + (1 × N) → (M × N)
    ///
    /// # Example
    ///
    /// ```rust
    /// use compositional_diff::tensor::Tensor;
    ///
    /// let x: Tensor<f32, 3, 4> = Tensor::zeros();           // 3×4
    /// let bias: Tensor<f32, 1, 4> = Tensor::zeros();        // 1×4 bias
    /// let y: Tensor<f32, 3, 4> = x.add_bias(&bias);         // broadcasts
    /// ```
    pub fn add_bias(&self, bias: &Tensor<T, 1, N>) -> Tensor<T, M, N> {
        let mut result = Tensor::zeros();
        for i in 0..M {
            for j in 0..N {
                result.data[i][j] = self.data[i][j] + bias.data[0][j];
            }
        }
        result
    }
}

impl<T, const M: usize, const N: usize> Tensor<T, M, N>
where
    T: Default + Copy + PartialOrd,
{
    /// Element-wise ReLU activation.
    ///
    /// ReLU(x) = max(0, x)
    ///
    /// # Example
    ///
    /// ```rust
    /// use compositional_diff::tensor::Tensor;
    ///
    /// let x: Tensor<f32, 2, 2> = Tensor::from_data([[-1.0, 2.0], [3.0, -4.0]]);
    /// let y = x.relu();
    /// assert_eq!(y.get(0, 0), 0.0);  // -1 → 0
    /// assert_eq!(y.get(0, 1), 2.0);  // 2 → 2
    /// ```
    pub fn relu(&self) -> Tensor<T, M, N> {
        let mut result = Tensor::zeros();
        let zero = T::default();
        for i in 0..M {
            for j in 0..N {
                result.data[i][j] = if self.data[i][j] > zero {
                    self.data[i][j]
                } else {
                    zero
                };
            }
        }
        result
    }
}

impl<T: Copy, const M: usize, const N: usize> Tensor<T, M, N> {
    /// Transpose the tensor.
    ///
    /// (M × N) → (N × M)
    pub fn transpose(&self) -> Tensor<T, N, M>
    where
        T: Default,
    {
        let mut result = Tensor::zeros();
        for i in 0..M {
            for j in 0..N {
                result.data[j][i] = self.data[i][j];
            }
        }
        result
    }
}

// ============================================================================
// Random Initialization
// ============================================================================

impl<const M: usize, const N: usize> Tensor<f32, M, N> {
    /// Create a tensor with simple pseudo-random values.
    ///
    /// Uses a basic LCG for reproducible "random" initialization.
    /// For real training, use a proper random number generator.
    pub fn random_seed(seed: u64) -> Self {
        let mut result = Self::zeros();
        let mut state = seed;
        for i in 0..M {
            for j in 0..N {
                // Simple LCG: x_{n+1} = (a * x_n + c) mod m
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                // Scale to [-1, 1]
                let val = ((state >> 33) as f32 / (u32::MAX as f32)) * 2.0 - 1.0;
                result.data[i][j] = val * 0.1; // Small initial values
            }
        }
        result
    }
}

// ============================================================================
// Debug/Display
// ============================================================================

impl<T: fmt::Debug, const M: usize, const N: usize> fmt::Debug for Tensor<T, M, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor<{}, {}> {:?}", M, N, self.data)
    }
}

// ============================================================================
// Neural Network Layers
// ============================================================================

/// A linear layer with compile-time dimension checking.
///
/// Computes: `output = input @ weights.T + bias`
///
/// # Type Parameters
///
/// - `IN`: Input dimension (const generic)
/// - `OUT`: Output dimension (const generic)
///
/// # Example
///
/// ```rust
/// use compositional_diff::tensor::{Tensor, Linear};
///
/// let layer: Linear<784, 256> = Linear::random();
///
/// // Single input: (1 × 784) → (1 × 256)
/// let input: Tensor<f32, 1, 784> = Tensor::zeros();
/// let output: Tensor<f32, 1, 256> = layer.forward(&input);
///
/// // Batched input: (32 × 784) → (32 × 256)
/// let batch: Tensor<f32, 32, 784> = Tensor::zeros();
/// let batch_out: Tensor<f32, 32, 256> = layer.forward(&batch);
/// ```
#[derive(Clone)]
pub struct Linear<const IN: usize, const OUT: usize> {
    /// Weights: OUT × IN (stored for efficient matmul with input)
    pub weights: Tensor<f32, OUT, IN>,
    /// Bias: 1 × OUT
    pub bias: Tensor<f32, 1, OUT>,
}

impl<const IN: usize, const OUT: usize> Linear<IN, OUT> {
    /// Create a new linear layer with zero weights and bias.
    pub fn zeros() -> Self {
        Self {
            weights: Tensor::zeros(),
            bias: Tensor::zeros(),
        }
    }

    /// Create a new linear layer with pseudo-random initialization.
    pub fn random() -> Self {
        Self {
            // Use different seeds for weights and bias
            weights: Tensor::random_seed(IN as u64 * 1000 + OUT as u64),
            bias: Tensor::zeros(),
        }
    }

    /// Forward pass with compile-time batch size checking.
    ///
    /// input: (BATCH × IN) → output: (BATCH × OUT)
    pub fn forward<const BATCH: usize>(
        &self,
        input: &Tensor<f32, BATCH, IN>,
    ) -> Tensor<f32, BATCH, OUT> {
        // input: (BATCH × IN) @ weights.T: (IN × OUT) → (BATCH × OUT)
        let weights_t = self.weights.transpose();
        input.matmul(&weights_t).add_bias(&self.bias)
    }
}

impl<const IN: usize, const OUT: usize> fmt::Debug for Linear<IN, OUT> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Linear<{}, {}>", IN, OUT)
    }
}

/// A multi-layer perceptron with one hidden layer.
///
/// Architecture: Input → Linear → ReLU → Linear → Output
///
/// # Type Parameters
///
/// - `IN`: Input dimension
/// - `HIDDEN`: Hidden layer dimension
/// - `OUT`: Output dimension
///
/// # Example
///
/// ```rust
/// use compositional_diff::tensor::{Tensor, MLP};
///
/// // Define architecture at compile time
/// let mlp: MLP<784, 256, 10> = MLP::random();
///
/// // Input shape is checked at compile time
/// let input: Tensor<f32, 1, 784> = Tensor::zeros();
/// let output: Tensor<f32, 1, 10> = mlp.forward(&input);
///
/// // This won't compile - wrong input dimension!
/// // let bad_input: Tensor<f32, 1, 100> = Tensor::zeros();
/// // let _ = mlp.forward(&bad_input);
/// ```
#[derive(Clone)]
pub struct MLP<const IN: usize, const HIDDEN: usize, const OUT: usize> {
    /// First layer: IN → HIDDEN
    pub layer1: Linear<IN, HIDDEN>,
    /// Second layer: HIDDEN → OUT
    pub layer2: Linear<HIDDEN, OUT>,
}

impl<const IN: usize, const HIDDEN: usize, const OUT: usize> MLP<IN, HIDDEN, OUT> {
    /// Create a new MLP with zero weights.
    pub fn zeros() -> Self {
        Self {
            layer1: Linear::zeros(),
            layer2: Linear::zeros(),
        }
    }

    /// Create a new MLP with pseudo-random initialization.
    pub fn random() -> Self {
        Self {
            layer1: Linear::random(),
            layer2: Linear::random(),
        }
    }

    /// Forward pass through the entire network.
    ///
    /// input: (BATCH × IN) → output: (BATCH × OUT)
    pub fn forward<const BATCH: usize>(
        &self,
        input: &Tensor<f32, BATCH, IN>,
    ) -> Tensor<f32, BATCH, OUT> {
        let h = self.layer1.forward(input).relu();
        self.layer2.forward(&h)
    }
}

impl<const IN: usize, const HIDDEN: usize, const OUT: usize> fmt::Debug for MLP<IN, HIDDEN, OUT> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MLP<{}, {}, {}>", IN, HIDDEN, OUT)
    }
}

// ============================================================================
// Categorical Composition Helpers
// ============================================================================

/// Compose two functions with type-checked intermediate dimensions.
///
/// This demonstrates that function composition in Rust is naturally
/// categorical - the type system enforces `cod(f) = dom(g)`.
///
/// # Example
///
/// ```rust
/// use compositional_diff::tensor::{Tensor, compose};
///
/// fn double(x: Tensor<f32, 1, 4>) -> Tensor<f32, 1, 4> {
///     x.add(&x)
/// }
///
/// fn extend(x: Tensor<f32, 1, 4>) -> Tensor<f32, 1, 8> {
///     Tensor::zeros() // placeholder
/// }
///
/// // f: (1×4) → (1×4)
/// // g: (1×4) → (1×8)
/// // f;g: (1×4) → (1×8)
/// let fg = compose(double, extend);
/// ```
pub fn compose<A, B, C, F, G>(f: F, g: G) -> impl Fn(A) -> C
where
    F: Fn(A) -> B,
    G: Fn(B) -> C,
{
    move |x| g(f(x))
}

/// Tensor product of two functions (parallel composition).
///
/// Given f: A → B and g: C → D, produces f⊗g: (A,C) → (B,D)
///
/// # Example
///
/// ```rust
/// use compositional_diff::tensor::{Tensor, tensor_product};
///
/// let f = |x: Tensor<f32, 1, 2>| x.relu();
/// let g = |x: Tensor<f32, 1, 3>| x.relu();
///
/// let fg = tensor_product(f, g);
/// let (a, b) = fg((Tensor::zeros(), Tensor::zeros()));
/// ```
pub fn tensor_product<A, B, C, D, F, G>(f: F, g: G) -> impl Fn((A, C)) -> (B, D)
where
    F: Fn(A) -> B,
    G: Fn(C) -> D,
{
    move |(a, c)| (f(a), g(c))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_zeros() {
        let t: Tensor<f32, 3, 4> = Tensor::zeros();
        assert_eq!(t.get(0, 0), 0.0);
        assert_eq!(t.get(2, 3), 0.0);
    }

    #[test]
    fn test_tensor_from_data() {
        let t: Tensor<i32, 2, 3> = Tensor::from_data([[1, 2, 3], [4, 5, 6]]);
        assert_eq!(t.get(0, 0), 1);
        assert_eq!(t.get(1, 2), 6);
    }

    #[test]
    fn test_matmul_dimensions() {
        let a: Tensor<f32, 2, 3> = Tensor::zeros();
        let b: Tensor<f32, 3, 4> = Tensor::zeros();
        let c: Tensor<f32, 2, 4> = a.matmul(&b);
        assert_eq!(Tensor::<f32, 2, 4>::ROWS, 2);
        assert_eq!(Tensor::<f32, 2, 4>::COLS, 4);
        assert_eq!(c.get(0, 0), 0.0);
    }

    #[test]
    fn test_matmul_values() {
        // [1 2] × [5 6]   [1*5+2*7  1*6+2*8]   [19 22]
        // [3 4]   [7 8] = [3*5+4*7  3*6+4*8] = [43 50]
        let a: Tensor<f32, 2, 2> = Tensor::from_data([[1.0, 2.0], [3.0, 4.0]]);
        let b: Tensor<f32, 2, 2> = Tensor::from_data([[5.0, 6.0], [7.0, 8.0]]);
        let c = a.matmul(&b);

        assert_eq!(c.get(0, 0), 19.0);
        assert_eq!(c.get(0, 1), 22.0);
        assert_eq!(c.get(1, 0), 43.0);
        assert_eq!(c.get(1, 1), 50.0);
    }

    #[test]
    fn test_add() {
        let a: Tensor<f32, 2, 2> = Tensor::from_data([[1.0, 2.0], [3.0, 4.0]]);
        let b: Tensor<f32, 2, 2> = Tensor::from_data([[10.0, 20.0], [30.0, 40.0]]);
        let c = a.add(&b);

        assert_eq!(c.get(0, 0), 11.0);
        assert_eq!(c.get(1, 1), 44.0);
    }

    #[test]
    fn test_relu() {
        let x: Tensor<f32, 2, 2> = Tensor::from_data([[-1.0, 2.0], [3.0, -4.0]]);
        let y = x.relu();

        assert_eq!(y.get(0, 0), 0.0);
        assert_eq!(y.get(0, 1), 2.0);
        assert_eq!(y.get(1, 0), 3.0);
        assert_eq!(y.get(1, 1), 0.0);
    }

    #[test]
    fn test_transpose() {
        let a: Tensor<f32, 2, 3> = Tensor::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let b: Tensor<f32, 3, 2> = a.transpose();

        assert_eq!(b.get(0, 0), 1.0);
        assert_eq!(b.get(0, 1), 4.0);
        assert_eq!(b.get(2, 0), 3.0);
        assert_eq!(b.get(2, 1), 6.0);
    }

    #[test]
    fn test_linear_layer() {
        let layer: Linear<4, 2> = Linear::zeros();
        let input: Tensor<f32, 1, 4> = Tensor::zeros();
        let output: Tensor<f32, 1, 2> = layer.forward(&input);
        assert_eq!(output.get(0, 0), 0.0);
    }

    #[test]
    fn test_linear_batched() {
        let layer: Linear<4, 2> = Linear::zeros();
        let batch: Tensor<f32, 32, 4> = Tensor::zeros();
        let output: Tensor<f32, 32, 2> = layer.forward(&batch);
        assert_eq!(Tensor::<f32, 32, 2>::ROWS, 32);
        assert_eq!(output.get(0, 0), 0.0);
    }

    #[test]
    fn test_mlp_forward() {
        let mlp: MLP<4, 8, 2> = MLP::zeros();
        let input: Tensor<f32, 1, 4> = Tensor::zeros();
        let output: Tensor<f32, 1, 2> = mlp.forward(&input);
        assert_eq!(output.get(0, 0), 0.0);
    }

    #[test]
    fn test_compose() {
        let f = |x: Tensor<f32, 1, 2>| x.add(&x);
        let g = |x: Tensor<f32, 1, 2>| x.relu();
        let fg = compose(f, g);

        let input: Tensor<f32, 1, 2> = Tensor::from_data([[-1.0, 2.0]]);
        let output = fg(input);

        // f: [-1, 2] → [-2, 4]
        // g: [-2, 4] → [0, 4]
        assert_eq!(output.get(0, 0), 0.0);
        assert_eq!(output.get(0, 1), 4.0);
    }

    #[test]
    fn test_tensor_product() {
        let f = |x: Tensor<f32, 1, 2>| x.relu();
        let g = |x: Tensor<f32, 1, 3>| x.relu();
        let fg = tensor_product(f, g);

        let a: Tensor<f32, 1, 2> = Tensor::from_data([[-1.0, 2.0]]);
        let b: Tensor<f32, 1, 3> = Tensor::from_data([[-1.0, 0.0, 1.0]]);
        let (ra, rb) = fg((a, b));

        assert_eq!(ra.get(0, 0), 0.0);
        assert_eq!(ra.get(0, 1), 2.0);
        assert_eq!(rb.get(0, 0), 0.0);
        assert_eq!(rb.get(0, 2), 1.0);
    }
}
