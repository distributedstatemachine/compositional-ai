//! Markov kernels (stochastic maps) between finite sets.

use crate::dist::Dist;
use crate::error::ProbError;
use crate::PROB_TOLERANCE;

/// A Markov kernel (stochastic map) from a finite set X to a finite set Y.
///
/// Represented as a row-stochastic matrix where:
/// - `k[i][j]` = P(output = j | input = i)
/// - Each row sums to 1
///
/// # Categorical View
///
/// A kernel K: n → m is a morphism in the category FinStoch.
/// - Objects are natural numbers (cardinalities)
/// - Composition is matrix multiplication
/// - Identity is the identity matrix
///
/// # Example
///
/// ```rust
/// use compositional_prob::{Kernel, Dist};
///
/// // Binary symmetric channel with error rate 0.1
/// let channel = Kernel::new(vec![
///     vec![0.9, 0.1],  // P(output | input=0)
///     vec![0.1, 0.9],  // P(output | input=1)
/// ]).unwrap();
///
/// // Send a uniform distribution through the channel
/// let input = Dist::uniform(2);
/// let output = channel.apply(&input).unwrap();
/// // Output is still uniform (symmetric channel preserves uniform)
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Kernel {
    /// Row-stochastic matrix: `k[i][j]` = P(output=j | input=i)
    pub k: Vec<Vec<f32>>,
    /// Number of input states
    pub n_inputs: usize,
    /// Number of output states
    pub n_outputs: usize,
}

impl Kernel {
    /// Create a new kernel from a row-stochastic matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The matrix is empty
    /// - Rows have different lengths
    /// - Any row doesn't sum to 1 (within tolerance)
    pub fn new(k: Vec<Vec<f32>>) -> Result<Self, ProbError> {
        if k.is_empty() {
            return Err(ProbError::EmptyKernel);
        }

        let n_inputs = k.len();
        let n_outputs = k[0].len();

        if n_outputs == 0 {
            return Err(ProbError::EmptyKernel);
        }

        // Verify each row is a valid probability distribution
        for (i, row) in k.iter().enumerate() {
            if row.len() != n_outputs {
                return Err(ProbError::RaggedMatrix);
            }

            // Check for negative probabilities
            if row.iter().any(|&x| x < -PROB_TOLERANCE) {
                return Err(ProbError::NegativeProbability);
            }

            // Check normalization
            let sum: f32 = row.iter().sum();
            if (sum - 1.0).abs() > PROB_TOLERANCE {
                return Err(ProbError::RowNotNormalized { row: i, sum });
            }
        }

        Ok(Self {
            k,
            n_inputs,
            n_outputs,
        })
    }

    /// Create the identity kernel: deterministically map i to i.
    ///
    /// This is the identity morphism in FinStoch.
    ///
    /// # Example
    ///
    /// ```rust
    /// use compositional_prob::{Kernel, Dist};
    ///
    /// let id = Kernel::identity(3);
    /// let d = Dist::new(vec![0.2, 0.3, 0.5]).unwrap();
    /// let result = id.apply(&d).unwrap();
    /// // Identity preserves the distribution
    /// assert!((result.p[0] - 0.2).abs() < 1e-6);
    /// ```
    pub fn identity(n: usize) -> Self {
        let k: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let mut row = vec![0.0; n];
                row[i] = 1.0;
                row
            })
            .collect();
        Self {
            k,
            n_inputs: n,
            n_outputs: n,
        }
    }

    /// Create a constant kernel: always output the same distribution.
    ///
    /// This ignores the input and always produces the given distribution.
    ///
    /// # Example
    ///
    /// ```rust
    /// use compositional_prob::{Kernel, Dist};
    ///
    /// let target = Dist::new(vec![0.3, 0.7]).unwrap();
    /// let k = Kernel::constant(5, &target);
    ///
    /// // Any input produces the same output
    /// let d = Dist::point(5, 2);
    /// let result = k.apply(&d).unwrap();
    /// assert!((result.p[0] - 0.3).abs() < 1e-6);
    /// ```
    pub fn constant(n_inputs: usize, dist: &Dist) -> Self {
        let k: Vec<Vec<f32>> = (0..n_inputs).map(|_| dist.p.clone()).collect();
        Self {
            k,
            n_inputs,
            n_outputs: dist.p.len(),
        }
    }

    /// Create a deterministic kernel from a function.
    ///
    /// Each input i maps to output f(i) with probability 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// use compositional_prob::Kernel;
    ///
    /// // Modulo 2 function: {0,1,2,3} -> {0,1}
    /// let k = Kernel::deterministic(4, 2, |i| i % 2);
    /// // Input 0 -> Output 0
    /// // Input 1 -> Output 1
    /// // Input 2 -> Output 0
    /// // Input 3 -> Output 1
    /// ```
    pub fn deterministic<F>(n_inputs: usize, n_outputs: usize, f: F) -> Self
    where
        F: Fn(usize) -> usize,
    {
        let k: Vec<Vec<f32>> = (0..n_inputs)
            .map(|i| {
                let j = f(i);
                assert!(
                    j < n_outputs,
                    "Function output {} >= n_outputs {}",
                    j,
                    n_outputs
                );
                let mut row = vec![0.0; n_outputs];
                row[j] = 1.0;
                row
            })
            .collect();
        Self {
            k,
            n_inputs,
            n_outputs,
        }
    }

    /// Create a uniform kernel: each input maps to uniform distribution over outputs.
    pub fn uniform(n_inputs: usize, n_outputs: usize) -> Self {
        let row = vec![1.0 / n_outputs as f32; n_outputs];
        let k: Vec<Vec<f32>> = (0..n_inputs).map(|_| row.clone()).collect();
        Self {
            k,
            n_inputs,
            n_outputs,
        }
    }

    /// Compose two kernels: self ; other.
    ///
    /// If self: n → m and other: m → p, then (self ; other): n → p.
    ///
    /// This is matrix multiplication, but categorically it's composition:
    /// (K ; L)(·|x) = Σᵧ K(y|x) · L(·|y)
    ///
    /// # Errors
    ///
    /// Returns an error if shapes don't match (self.n_outputs ≠ other.n_inputs).
    ///
    /// # Example
    ///
    /// ```rust
    /// use compositional_prob::Kernel;
    ///
    /// // Weather: Sunny=0, Rainy=1
    /// let weather = Kernel::new(vec![
    ///     vec![0.8, 0.2],  // Sunny -> 80% Sunny, 20% Rainy
    ///     vec![0.4, 0.6],  // Rainy -> 40% Sunny, 60% Rainy
    /// ]).unwrap();
    ///
    /// // Two-day transition
    /// let two_days = weather.compose(&weather).unwrap();
    /// ```
    pub fn compose(&self, other: &Kernel) -> Result<Kernel, ProbError> {
        if self.n_outputs != other.n_inputs {
            return Err(ProbError::ShapeMismatch {
                expected: self.n_outputs,
                got: other.n_inputs,
            });
        }

        let n = self.n_inputs;
        let m = self.n_outputs;
        let p = other.n_outputs;

        let mut result = vec![vec![0.0; p]; n];

        for (i, result_row) in result.iter_mut().enumerate() {
            for (k, result_elem) in result_row.iter_mut().enumerate() {
                let mut sum = 0.0;
                for j in 0..m {
                    sum += self.k[i][j] * other.k[j][k];
                }
                *result_elem = sum;
            }
        }

        // Result is guaranteed to be row-stochastic
        Ok(Kernel {
            k: result,
            n_inputs: n,
            n_outputs: p,
        })
    }

    /// Apply the kernel to a distribution.
    ///
    /// If K: n → m and p: Dist(n), then K(p): Dist(m).
    ///
    /// `K(p)[j] = Σᵢ p[i] · K[i,j]`
    ///
    /// This is vector-matrix multiplication.
    ///
    /// # Errors
    ///
    /// Returns an error if the distribution size doesn't match n_inputs.
    pub fn apply(&self, dist: &Dist) -> Result<Dist, ProbError> {
        if dist.p.len() != self.n_inputs {
            return Err(ProbError::ShapeMismatch {
                expected: self.n_inputs,
                got: dist.p.len(),
            });
        }

        let mut result = vec![0.0; self.n_outputs];
        for (j, result_elem) in result.iter_mut().enumerate() {
            for i in 0..self.n_inputs {
                *result_elem += dist.p[i] * self.k[i][j];
            }
        }

        // Result is guaranteed to be a valid distribution
        Ok(Dist { p: result })
    }

    /// Apply the kernel to a single input state (deterministic input).
    ///
    /// Returns the i-th row as a distribution.
    pub fn apply_to_state(&self, i: usize) -> Result<Dist, ProbError> {
        if i >= self.n_inputs {
            return Err(ProbError::IndexOutOfBounds {
                index: i,
                size: self.n_inputs,
            });
        }
        Ok(Dist {
            p: self.k[i].clone(),
        })
    }

    /// Get a specific conditional probability P(output=j | input=i).
    pub fn conditional(&self, input: usize, output: usize) -> Result<f32, ProbError> {
        if input >= self.n_inputs {
            return Err(ProbError::IndexOutOfBounds {
                index: input,
                size: self.n_inputs,
            });
        }
        if output >= self.n_outputs {
            return Err(ProbError::IndexOutOfBounds {
                index: output,
                size: self.n_outputs,
            });
        }
        Ok(self.k[input][output])
    }

    /// Check if the kernel is deterministic (each row is a point mass).
    pub fn is_deterministic(&self) -> bool {
        for row in &self.k {
            let num_nonzero = row.iter().filter(|&&x| x > PROB_TOLERANCE).count();
            if num_nonzero != 1 {
                return false;
            }
        }
        true
    }

    /// Check if the kernel is doubly stochastic (columns also sum to 1).
    ///
    /// Doubly stochastic matrices preserve the uniform distribution.
    pub fn is_doubly_stochastic(&self) -> bool {
        if self.n_inputs != self.n_outputs {
            return false;
        }

        for j in 0..self.n_outputs {
            let col_sum: f32 = (0..self.n_inputs).map(|i| self.k[i][j]).sum();
            if (col_sum - 1.0).abs() > PROB_TOLERANCE {
                return false;
            }
        }
        true
    }

    /// Compute the stationary distribution (if unique).
    ///
    /// A stationary distribution π satisfies: π · K = π
    ///
    /// Uses power iteration. Returns None if not converged.
    pub fn stationary(&self, max_iter: usize, tolerance: f32) -> Option<Dist> {
        if self.n_inputs != self.n_outputs {
            return None;
        }

        let n = self.n_inputs;
        let mut dist = Dist::uniform(n);

        for _ in 0..max_iter {
            let next = self.apply(&dist).ok()?;

            // Check convergence
            let diff: f32 = dist
                .p
                .iter()
                .zip(next.p.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();

            if diff < tolerance {
                return Some(next);
            }

            dist = next;
        }

        None // Didn't converge
    }

    /// Tensor product of two kernels.
    ///
    /// If K: n → m and L: p → q, then K ⊗ L: n*p → m*q.
    ///
    /// (K ⊗ L)[(i1, i2), (j1, j2)] = K[i1, j1] · L[i2, j2]
    pub fn tensor(&self, other: &Kernel) -> Kernel {
        let n_in = self.n_inputs * other.n_inputs;
        let n_out = self.n_outputs * other.n_outputs;

        let mut k = vec![vec![0.0; n_out]; n_in];

        for i1 in 0..self.n_inputs {
            for i2 in 0..other.n_inputs {
                let i = i1 * other.n_inputs + i2;
                for j1 in 0..self.n_outputs {
                    for j2 in 0..other.n_outputs {
                        let j = j1 * other.n_outputs + j2;
                        k[i][j] = self.k[i1][j1] * other.k[i2][j2];
                    }
                }
            }
        }

        Kernel {
            k,
            n_inputs: n_in,
            n_outputs: n_out,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_new_valid() {
        let k = Kernel::new(vec![vec![0.3, 0.7], vec![0.5, 0.5]]).unwrap();
        assert_eq!(k.n_inputs, 2);
        assert_eq!(k.n_outputs, 2);
    }

    #[test]
    fn test_kernel_new_not_normalized() {
        let result = Kernel::new(vec![vec![0.3, 0.6], vec![0.5, 0.5]]);
        assert!(matches!(result, Err(ProbError::RowNotNormalized { .. })));
    }

    #[test]
    fn test_kernel_identity() {
        let id = Kernel::identity(3);
        assert_eq!(id.k[0], vec![1.0, 0.0, 0.0]);
        assert_eq!(id.k[1], vec![0.0, 1.0, 0.0]);
        assert_eq!(id.k[2], vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_kernel_apply() {
        let k = Kernel::new(vec![vec![0.9, 0.1], vec![0.1, 0.9]]).unwrap();

        // Apply to point mass at 0
        let d = Dist::point(2, 0);
        let result = k.apply(&d).unwrap();
        assert!((result.p[0] - 0.9).abs() < 1e-6);
        assert!((result.p[1] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_kernel_compose() {
        let k1 = Kernel::new(vec![vec![0.5, 0.5], vec![0.3, 0.7]]).unwrap();

        let k2 = Kernel::new(vec![vec![0.6, 0.4], vec![0.2, 0.8]]).unwrap();

        let composed = k1.compose(&k2).unwrap();

        // Manual calculation for k1[0] ; k2:
        // [0, 0] = 0.5 * 0.6 + 0.5 * 0.2 = 0.4
        // [0, 1] = 0.5 * 0.4 + 0.5 * 0.8 = 0.6
        assert!((composed.k[0][0] - 0.4).abs() < 1e-6);
        assert!((composed.k[0][1] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_composition_preserves_stochastic() {
        let k1 = Kernel::new(vec![vec![0.5, 0.5], vec![0.3, 0.7]]).unwrap();

        let k2 = Kernel::new(vec![vec![0.6, 0.4], vec![0.2, 0.8]]).unwrap();

        let composed = k1.compose(&k2).unwrap();

        // Each row should still sum to 1
        for row in &composed.k {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < PROB_TOLERANCE);
        }
    }

    #[test]
    fn test_identity_composition() {
        let k = Kernel::new(vec![vec![0.3, 0.7], vec![0.5, 0.5]]).unwrap();

        let id = Kernel::identity(2);

        // id ; k = k
        let composed_left = id.compose(&k).unwrap();
        assert!((composed_left.k[0][0] - k.k[0][0]).abs() < 1e-6);

        // k ; id = k
        let composed_right = k.compose(&id).unwrap();
        assert!((composed_right.k[0][0] - k.k[0][0]).abs() < 1e-6);
    }

    #[test]
    fn test_deterministic_kernel() {
        let k = Kernel::deterministic(3, 2, |i| i % 2);
        assert!(k.is_deterministic());
        assert_eq!(k.k[0], vec![1.0, 0.0]);
        assert_eq!(k.k[1], vec![0.0, 1.0]);
        assert_eq!(k.k[2], vec![1.0, 0.0]);
    }

    #[test]
    fn test_doubly_stochastic() {
        // Permutation matrix is doubly stochastic
        let k = Kernel::deterministic(3, 3, |i| (i + 1) % 3);
        assert!(k.is_doubly_stochastic());

        // Non-square is not doubly stochastic
        let k2 = Kernel::new(vec![vec![0.5, 0.5]]).unwrap();
        assert!(!k2.is_doubly_stochastic());
    }

    #[test]
    fn test_stationary_distribution() {
        // Regular Markov chain: all states reachable from each other
        let k = Kernel::new(vec![vec![0.9, 0.1], vec![0.2, 0.8]]).unwrap();

        let stationary = k.stationary(1000, 1e-6).unwrap();

        // Verify π · K = π
        let after = k.apply(&stationary).unwrap();
        for i in 0..2 {
            assert!((stationary.p[i] - after.p[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_tensor_product() {
        let k1 = Kernel::identity(2);
        let k2 = Kernel::identity(2);
        let product = k1.tensor(&k2);

        // Identity ⊗ Identity = Identity on product space
        assert_eq!(product.n_inputs, 4);
        assert_eq!(product.n_outputs, 4);
        assert!(product.is_deterministic());
    }

    #[test]
    fn test_weather_example() {
        // Weather Markov chain from notes
        let weather = Kernel::new(vec![
            vec![0.8, 0.2], // Sunny -> 80% Sunny, 20% Rainy
            vec![0.4, 0.6], // Rainy -> 40% Sunny, 60% Rainy
        ])
        .unwrap();

        // Start sunny
        let today = Dist::point(2, 0);

        // Tomorrow
        let tomorrow = weather.apply(&today).unwrap();
        assert!((tomorrow.p[0] - 0.8).abs() < 1e-6);
        assert!((tomorrow.p[1] - 0.2).abs() < 1e-6);

        // Find stationary distribution
        let stationary = weather.stationary(100, 1e-6).unwrap();
        // Should be [2/3, 1/3] (from solving π = π·K)
        assert!((stationary.p[0] - 2.0 / 3.0).abs() < 1e-4);
        assert!((stationary.p[1] - 1.0 / 3.0).abs() < 1e-4);
    }
}
