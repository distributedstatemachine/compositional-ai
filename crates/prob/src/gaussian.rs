//! Continuous Gaussian distributions and kernels.
//!
//! This module implements the category **Gauss** where:
//! - Objects are dimensions (natural numbers representing ℝⁿ)
//! - Morphisms are Gaussian channels (linear-Gaussian kernels)
//!
//! The key insight is that composition of Gaussian kernels follows the
//! **convolution formula**, which is central to Kalman filtering, VAEs,
//! and many other ML algorithms.
//!
//! # Example
//!
//! ```rust
//! use compositional_prob::gaussian::{GaussianDist, GaussianKernel};
//!
//! // A 1D Gaussian: N(0, 1)
//! let standard_normal = GaussianDist::new(0.0, 1.0).unwrap();
//!
//! // A linear Gaussian kernel: x ↦ N(2x + 1, 0.5²)
//! let kernel = GaussianKernel::linear(2.0, 1.0, 0.5);
//!
//! // Apply to a point
//! let output = kernel.apply(3.0);  // N(2*3 + 1, 0.25) = N(7, 0.25)
//! assert!((output.mean - 7.0).abs() < 1e-6);
//! ```

use crate::ProbError;
use std::f64::consts::{E, PI};

/// A univariate Gaussian (Normal) distribution.
///
/// Represents N(μ, σ²) where:
/// - μ (mean) is the center of the distribution
/// - σ² (variance) measures the spread (must be non-negative)
/// - σ (std dev) is √σ²
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianDist {
    /// Mean (μ)
    pub mean: f64,
    /// Standard deviation (σ)
    pub std_dev: f64,
}

impl GaussianDist {
    /// Create a new Gaussian distribution N(mean, std_dev²).
    ///
    /// # Arguments
    /// * `mean` - The mean μ
    /// * `std_dev` - The standard deviation σ (must be non-negative)
    ///
    /// # Errors
    /// Returns error if std_dev is negative.
    pub fn new(mean: f64, std_dev: f64) -> Result<Self, ProbError> {
        if std_dev < 0.0 {
            return Err(ProbError::InvalidParameter {
                name: "std_dev".to_string(),
                reason: "standard deviation must be non-negative".to_string(),
            });
        }
        Ok(Self { mean, std_dev })
    }

    /// Create the standard normal distribution N(0, 1).
    pub fn standard() -> Self {
        Self {
            mean: 0.0,
            std_dev: 1.0,
        }
    }

    /// Create a point mass (degenerate Gaussian with σ = 0).
    pub fn point(value: f64) -> Self {
        Self {
            mean: value,
            std_dev: 0.0,
        }
    }

    /// The variance σ².
    pub fn variance(&self) -> f64 {
        self.std_dev * self.std_dev
    }

    /// The precision τ = 1/σ² (inverse variance).
    ///
    /// Returns infinity for point masses.
    pub fn precision(&self) -> f64 {
        if self.std_dev == 0.0 {
            f64::INFINITY
        } else {
            1.0 / self.variance()
        }
    }

    /// Probability density function at x.
    ///
    /// pdf(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))
    pub fn pdf(&self, x: f64) -> f64 {
        if self.std_dev == 0.0 {
            // Point mass: pdf is infinity at mean, 0 elsewhere
            if (x - self.mean).abs() < 1e-10 {
                f64::INFINITY
            } else {
                0.0
            }
        } else {
            let z = (x - self.mean) / self.std_dev;
            let normalization = 1.0 / (self.std_dev * (2.0 * PI).sqrt());
            normalization * (-0.5 * z * z).exp()
        }
    }

    /// Log probability density at x.
    ///
    /// log pdf(x) = -0.5 * log(2πσ²) - (x-μ)²/(2σ²)
    pub fn log_pdf(&self, x: f64) -> f64 {
        if self.std_dev == 0.0 {
            if (x - self.mean).abs() < 1e-10 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            }
        } else {
            let z = (x - self.mean) / self.std_dev;
            -0.5 * (2.0 * PI).ln() - self.std_dev.ln() - 0.5 * z * z
        }
    }

    /// Cumulative distribution function (using error function approximation).
    ///
    /// CDF(x) = P(X ≤ x) = 0.5 * (1 + erf((x-μ)/(σ√2)))
    pub fn cdf(&self, x: f64) -> f64 {
        if self.std_dev == 0.0 {
            if x < self.mean {
                0.0
            } else {
                1.0
            }
        } else {
            let z = (x - self.mean) / (self.std_dev * 2.0_f64.sqrt());
            0.5 * (1.0 + erf(z))
        }
    }

    /// Differential entropy: H(X) = 0.5 * log(2πeσ²)
    ///
    /// Returns negative infinity for point masses.
    pub fn entropy(&self) -> f64 {
        if self.std_dev == 0.0 {
            f64::NEG_INFINITY
        } else {
            0.5 * (2.0 * PI * E * self.variance()).ln()
        }
    }

    /// KL divergence from self to other: D_KL(self || other).
    ///
    /// D_KL(N(μ₁,σ₁²) || N(μ₂,σ₂²)) = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 0.5
    pub fn kl_divergence(&self, other: &GaussianDist) -> Result<f64, ProbError> {
        if other.std_dev == 0.0 {
            // KL to a point mass is infinity unless we're the same point
            if self.std_dev == 0.0 && (self.mean - other.mean).abs() < 1e-10 {
                return Ok(0.0);
            }
            return Ok(f64::INFINITY);
        }

        if self.std_dev == 0.0 {
            // Point mass to non-point: finite KL
            let diff = self.mean - other.mean;
            return Ok(diff * diff / (2.0 * other.variance()) + other.std_dev.ln() + 0.5);
        }

        let var1 = self.variance();
        let var2 = other.variance();
        let diff = self.mean - other.mean;

        Ok((other.std_dev / self.std_dev).ln() + (var1 + diff * diff) / (2.0 * var2) - 0.5)
    }

    /// Sample using the reparameterization trick.
    ///
    /// Given ε ~ N(0,1), returns μ + σ*ε.
    ///
    /// This is crucial for VAEs as it allows gradients to flow through sampling.
    pub fn sample_reparam(&self, epsilon: f64) -> f64 {
        self.mean + self.std_dev * epsilon
    }

    /// Inverse reparameterization: recover ε from sample x.
    ///
    /// ε = (x - μ) / σ
    pub fn reparam_inverse(&self, x: f64) -> Result<f64, ProbError> {
        if self.std_dev == 0.0 {
            if (x - self.mean).abs() < 1e-10 {
                Ok(0.0)
            } else {
                Err(ProbError::InvalidParameter {
                    name: "x".to_string(),
                    reason: "sample outside support of point mass".to_string(),
                })
            }
        } else {
            Ok((x - self.mean) / self.std_dev)
        }
    }

    /// Sum of two independent Gaussians.
    ///
    /// If X ~ N(μ₁, σ₁²) and Y ~ N(μ₂, σ₂²) are independent,
    /// then X + Y ~ N(μ₁ + μ₂, σ₁² + σ₂²).
    pub fn convolve(&self, other: &GaussianDist) -> GaussianDist {
        GaussianDist {
            mean: self.mean + other.mean,
            std_dev: (self.variance() + other.variance()).sqrt(),
        }
    }

    /// Scale a Gaussian by a constant.
    ///
    /// If X ~ N(μ, σ²), then aX ~ N(aμ, a²σ²).
    pub fn scale(&self, a: f64) -> GaussianDist {
        GaussianDist {
            mean: a * self.mean,
            std_dev: a.abs() * self.std_dev,
        }
    }

    /// Shift a Gaussian by a constant.
    ///
    /// If X ~ N(μ, σ²), then X + b ~ N(μ + b, σ²).
    pub fn shift(&self, b: f64) -> GaussianDist {
        GaussianDist {
            mean: self.mean + b,
            std_dev: self.std_dev,
        }
    }
}

/// A 1D linear Gaussian kernel (morphism in the category Gauss).
///
/// Represents the stochastic map x ↦ N(ax + b, σ²) where:
/// - a is the scale factor
/// - b is the bias/offset
/// - σ² is the noise variance
///
/// # Composition
///
/// If K₁: x ↦ N(a₁x + b₁, σ₁²) and K₂: y ↦ N(a₂y + b₂, σ₂²), then:
///
/// K₂ ∘ K₁: x ↦ N(a₂a₁x + a₂b₁ + b₂, a₂²σ₁² + σ₂²)
///
/// This is the **convolution formula** for Gaussian kernels.
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianKernel {
    /// Scale factor (a)
    pub scale: f64,
    /// Bias/offset (b)
    pub bias: f64,
    /// Noise standard deviation (σ)
    pub noise_std: f64,
}

impl GaussianKernel {
    /// Create a linear Gaussian kernel: x ↦ N(scale*x + bias, noise_std²).
    pub fn linear(scale: f64, bias: f64, noise_std: f64) -> Self {
        assert!(
            noise_std >= 0.0,
            "noise standard deviation must be non-negative"
        );
        Self {
            scale,
            bias,
            noise_std,
        }
    }

    /// Create the identity kernel: x ↦ N(x, 0) (deterministic identity).
    pub fn identity() -> Self {
        Self {
            scale: 1.0,
            bias: 0.0,
            noise_std: 0.0,
        }
    }

    /// Create a constant kernel: x ↦ N(μ, σ²) (ignores input).
    pub fn constant(mean: f64, std_dev: f64) -> Self {
        Self {
            scale: 0.0,
            bias: mean,
            noise_std: std_dev,
        }
    }

    /// Create a pure noise kernel: x ↦ N(x, σ²) (adds noise).
    pub fn add_noise(noise_std: f64) -> Self {
        Self {
            scale: 1.0,
            bias: 0.0,
            noise_std,
        }
    }

    /// The noise variance σ².
    pub fn noise_variance(&self) -> f64 {
        self.noise_std * self.noise_std
    }

    /// Apply the kernel to a point, returning the output distribution.
    ///
    /// x ↦ N(scale*x + bias, noise_std²)
    pub fn apply(&self, x: f64) -> GaussianDist {
        GaussianDist {
            mean: self.scale * x + self.bias,
            std_dev: self.noise_std,
        }
    }

    /// Apply the kernel to an input distribution.
    ///
    /// If input ~ N(μ_in, σ_in²), output ~ N(scale*μ_in + bias, scale²*σ_in² + noise_std²)
    pub fn apply_dist(&self, input: &GaussianDist) -> GaussianDist {
        let output_mean = self.scale * input.mean + self.bias;
        let output_var = self.scale * self.scale * input.variance() + self.noise_variance();
        GaussianDist {
            mean: output_mean,
            std_dev: output_var.sqrt(),
        }
    }

    /// Compose two kernels: self ∘ other (first apply other, then self).
    ///
    /// If K₁: x ↦ N(a₁x + b₁, σ₁²) and K₂: y ↦ N(a₂y + b₂, σ₂²), then:
    /// K₂ ∘ K₁: x ↦ N(a₂a₁x + a₂b₁ + b₂, a₂²σ₁² + σ₂²)
    ///
    /// Note: This is self ∘ other, so self is K₂ and other is K₁.
    pub fn compose(&self, other: &GaussianKernel) -> GaussianKernel {
        // self is K₂, other is K₁
        // K₂ ∘ K₁: x ↦ N(a₂*a₁*x + a₂*b₁ + b₂, a₂²*σ₁² + σ₂²)
        let new_scale = self.scale * other.scale;
        let new_bias = self.scale * other.bias + self.bias;
        let new_noise_var =
            self.scale * self.scale * other.noise_variance() + self.noise_variance();

        GaussianKernel {
            scale: new_scale,
            bias: new_bias,
            noise_std: new_noise_var.sqrt(),
        }
    }

    /// Sample from the kernel using the reparameterization trick.
    ///
    /// Given input x and ε ~ N(0,1), returns scale*x + bias + noise_std*ε.
    pub fn sample_reparam(&self, x: f64, epsilon: f64) -> f64 {
        self.scale * x + self.bias + self.noise_std * epsilon
    }

    /// The differential entropy added by this kernel.
    ///
    /// H(K) = 0.5 * log(2πe*noise_var) if noise_var > 0, else -∞
    pub fn entropy(&self) -> f64 {
        if self.noise_std == 0.0 {
            f64::NEG_INFINITY
        } else {
            0.5 * (2.0 * PI * E * self.noise_variance()).ln()
        }
    }
}

/// A multivariate Gaussian distribution N(μ, Σ).
///
/// Represents a distribution over ℝⁿ with:
/// - μ: mean vector (n-dimensional)
/// - Σ: covariance matrix (n×n, positive semi-definite)
#[derive(Debug, Clone, PartialEq)]
pub struct MultivariateGaussian {
    /// Mean vector μ
    pub mean: Vec<f64>,
    /// Covariance matrix Σ (stored as row-major flattened array)
    /// For an n-dimensional Gaussian, this has n² elements.
    pub covariance: Vec<f64>,
    /// Dimension n
    pub dim: usize,
}

impl MultivariateGaussian {
    /// Create a new multivariate Gaussian.
    ///
    /// # Arguments
    /// * `mean` - Mean vector μ (length n)
    /// * `covariance` - Covariance matrix Σ (n² elements, row-major)
    pub fn new(mean: Vec<f64>, covariance: Vec<f64>) -> Result<Self, ProbError> {
        let n = mean.len();
        if covariance.len() != n * n {
            return Err(ProbError::ShapeMismatch {
                expected: n * n,
                got: covariance.len(),
            });
        }

        // Check symmetry (approximately)
        for i in 0..n {
            for j in 0..i {
                let diff = (covariance[i * n + j] - covariance[j * n + i]).abs();
                if diff > 1e-10 {
                    return Err(ProbError::InvalidParameter {
                        name: "covariance".to_string(),
                        reason: "covariance matrix must be symmetric".to_string(),
                    });
                }
            }
        }

        // Check diagonal is non-negative
        for i in 0..n {
            if covariance[i * n + i] < -1e-10 {
                return Err(ProbError::InvalidParameter {
                    name: "covariance".to_string(),
                    reason: "diagonal elements must be non-negative".to_string(),
                });
            }
        }

        Ok(Self {
            mean,
            covariance,
            dim: n,
        })
    }

    /// Create a spherical Gaussian N(μ, σ²I).
    pub fn spherical(mean: Vec<f64>, variance: f64) -> Result<Self, ProbError> {
        if variance < 0.0 {
            return Err(ProbError::InvalidParameter {
                name: "variance".to_string(),
                reason: "variance must be non-negative".to_string(),
            });
        }
        let n = mean.len();
        let mut cov = vec![0.0; n * n];
        for i in 0..n {
            cov[i * n + i] = variance;
        }
        Ok(Self {
            mean,
            covariance: cov,
            dim: n,
        })
    }

    /// Create a diagonal Gaussian with specified variances.
    pub fn diagonal(mean: Vec<f64>, variances: Vec<f64>) -> Result<Self, ProbError> {
        let n = mean.len();
        if variances.len() != n {
            return Err(ProbError::ShapeMismatch {
                expected: n,
                got: variances.len(),
            });
        }
        for (i, &v) in variances.iter().enumerate() {
            if v < 0.0 {
                return Err(ProbError::InvalidParameter {
                    name: format!("variance[{}]", i),
                    reason: "variance must be non-negative".to_string(),
                });
            }
        }
        let mut cov = vec![0.0; n * n];
        for i in 0..n {
            cov[i * n + i] = variances[i];
        }
        Ok(Self {
            mean,
            covariance: cov,
            dim: n,
        })
    }

    /// Get element (i, j) of the covariance matrix.
    pub fn cov(&self, i: usize, j: usize) -> f64 {
        self.covariance[i * self.dim + j]
    }

    /// Get the standard deviation for dimension i.
    pub fn std_dev(&self, i: usize) -> f64 {
        self.cov(i, i).sqrt()
    }

    /// Get the marginal distribution for dimension i.
    pub fn marginal(&self, i: usize) -> GaussianDist {
        GaussianDist {
            mean: self.mean[i],
            std_dev: self.std_dev(i),
        }
    }

    /// Sample using reparameterization (diagonal case only for simplicity).
    ///
    /// For diagonal covariance: x = μ + σ ⊙ ε where ε ~ N(0, I)
    pub fn sample_reparam_diagonal(&self, epsilons: &[f64]) -> Result<Vec<f64>, ProbError> {
        if epsilons.len() != self.dim {
            return Err(ProbError::ShapeMismatch {
                expected: self.dim,
                got: epsilons.len(),
            });
        }

        let sample: Vec<f64> = epsilons
            .iter()
            .enumerate()
            .map(|(i, &eps)| self.mean[i] + self.std_dev(i) * eps)
            .collect();
        Ok(sample)
    }
}

/// A linear Gaussian kernel for multivariate distributions.
///
/// Represents the stochastic map x ↦ N(Ax + b, Σ) where:
/// - A is the transformation matrix (m×n)
/// - b is the bias vector (m-dimensional)
/// - Σ is the noise covariance (m×m)
#[derive(Debug, Clone)]
pub struct MultivariateGaussianKernel {
    /// Transformation matrix A (m×n, row-major)
    pub transform: Vec<f64>,
    /// Bias vector b (m-dimensional)
    pub bias: Vec<f64>,
    /// Noise covariance Σ (m×m, row-major)
    pub noise_cov: Vec<f64>,
    /// Input dimension n
    pub input_dim: usize,
    /// Output dimension m
    pub output_dim: usize,
}

impl MultivariateGaussianKernel {
    /// Create a new multivariate Gaussian kernel.
    pub fn new(
        transform: Vec<f64>,
        bias: Vec<f64>,
        noise_cov: Vec<f64>,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Self, ProbError> {
        if transform.len() != output_dim * input_dim {
            return Err(ProbError::ShapeMismatch {
                expected: output_dim * input_dim,
                got: transform.len(),
            });
        }
        if bias.len() != output_dim {
            return Err(ProbError::ShapeMismatch {
                expected: output_dim,
                got: bias.len(),
            });
        }
        if noise_cov.len() != output_dim * output_dim {
            return Err(ProbError::ShapeMismatch {
                expected: output_dim * output_dim,
                got: noise_cov.len(),
            });
        }

        Ok(Self {
            transform,
            bias,
            noise_cov,
            input_dim,
            output_dim,
        })
    }

    /// Create the identity kernel (n-dimensional).
    pub fn identity(dim: usize) -> Self {
        let mut transform = vec![0.0; dim * dim];
        for i in 0..dim {
            transform[i * dim + i] = 1.0;
        }

        Self {
            transform,
            bias: vec![0.0; dim],
            noise_cov: vec![0.0; dim * dim], // Zero noise (deterministic)
            input_dim: dim,
            output_dim: dim,
        }
    }

    /// Apply to a point x, returning the output distribution.
    pub fn apply(&self, x: &[f64]) -> Result<MultivariateGaussian, ProbError> {
        if x.len() != self.input_dim {
            return Err(ProbError::ShapeMismatch {
                expected: self.input_dim,
                got: x.len(),
            });
        }

        // Compute Ax + b
        let mean: Vec<f64> = (0..self.output_dim)
            .map(|i| {
                self.bias[i]
                    + x.iter()
                        .enumerate()
                        .map(|(j, &xj)| self.transform[i * self.input_dim + j] * xj)
                        .sum::<f64>()
            })
            .collect();

        Ok(MultivariateGaussian {
            mean,
            covariance: self.noise_cov.clone(),
            dim: self.output_dim,
        })
    }

    /// Compose two kernels: self ∘ other.
    ///
    /// If K₁: ℝⁿ → ℝᵐ with K₁(x) = N(A₁x + b₁, Σ₁)
    /// and K₂: ℝᵐ → ℝᵏ with K₂(y) = N(A₂y + b₂, Σ₂)
    /// then K₂∘K₁: ℝⁿ → ℝᵏ with (K₂∘K₁)(x) = N(A₂A₁x + A₂b₁ + b₂, A₂Σ₁A₂ᵀ + Σ₂)
    pub fn compose(&self, other: &MultivariateGaussianKernel) -> Result<Self, ProbError> {
        // self is K₂ (m → k), other is K₁ (n → m)
        if self.input_dim != other.output_dim {
            return Err(ProbError::ShapeMismatch {
                expected: self.input_dim,
                got: other.output_dim,
            });
        }

        let n = other.input_dim;
        let m = other.output_dim; // = self.input_dim
        let k = self.output_dim;

        // New transform: A₂ * A₁ (k×n matrix)
        let mut new_transform = vec![0.0; k * n];
        for i in 0..k {
            for j in 0..n {
                for l in 0..m {
                    new_transform[i * n + j] +=
                        self.transform[i * m + l] * other.transform[l * n + j];
                }
            }
        }

        // New bias: A₂ * b₁ + b₂
        let new_bias: Vec<f64> = self
            .bias
            .iter()
            .enumerate()
            .map(|(i, &b)| {
                b + other
                    .bias
                    .iter()
                    .enumerate()
                    .map(|(l, &bl)| self.transform[i * m + l] * bl)
                    .sum::<f64>()
            })
            .collect();

        // New noise covariance: A₂ * Σ₁ * A₂ᵀ + Σ₂
        // First compute A₂ * Σ₁ (k×m matrix)
        let mut a2_sigma1 = vec![0.0; k * m];
        for i in 0..k {
            for j in 0..m {
                for l in 0..m {
                    a2_sigma1[i * m + j] += self.transform[i * m + l] * other.noise_cov[l * m + j];
                }
            }
        }

        // Then compute (A₂ * Σ₁) * A₂ᵀ + Σ₂ (k×k matrix)
        let mut new_noise_cov = self.noise_cov.clone();
        for i in 0..k {
            for j in 0..k {
                for l in 0..m {
                    new_noise_cov[i * k + j] += a2_sigma1[i * m + l] * self.transform[j * m + l];
                }
            }
        }

        Ok(Self {
            transform: new_transform,
            bias: new_bias,
            noise_cov: new_noise_cov,
            input_dim: n,
            output_dim: k,
        })
    }
}

/// Error function (erf) approximation using Horner's method.
///
/// Accurate to about 1.5×10⁻⁷.
fn erf(x: f64) -> f64 {
    // Approximation from Abramowitz and Stegun
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// The category Gauss of Gaussian distributions.
///
/// - Objects: Natural numbers (dimensions of ℝⁿ)
/// - Morphisms: Linear Gaussian kernels
/// - Composition: Convolution formula
/// - Identity: Deterministic identity kernel
pub struct Gauss;

impl Gauss {
    /// Identity morphism for dimension n.
    pub fn id(n: usize) -> MultivariateGaussianKernel {
        MultivariateGaussianKernel::identity(n)
    }

    /// Compose two morphisms.
    pub fn compose(
        f: &MultivariateGaussianKernel,
        g: &MultivariateGaussianKernel,
    ) -> Result<MultivariateGaussianKernel, ProbError> {
        g.compose(f)
    }

    /// A distribution as a morphism from 1 → n.
    pub fn distribution(dist: &MultivariateGaussian) -> MultivariateGaussianKernel {
        // 1 → n: ignores input, outputs the distribution
        MultivariateGaussianKernel {
            transform: vec![0.0; dist.dim], // 0 matrix (n×1)
            bias: dist.mean.clone(),
            noise_cov: dist.covariance.clone(),
            input_dim: 1,
            output_dim: dist.dim,
        }
    }

    /// The unique morphism to the terminal object (n → 1).
    /// Marginalizes out all information.
    pub fn discard(n: usize) -> MultivariateGaussianKernel {
        MultivariateGaussianKernel {
            transform: vec![0.0; n], // 0 matrix (1×n)
            bias: vec![0.0],
            noise_cov: vec![0.0],
            input_dim: n,
            output_dim: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-6;

    #[test]
    fn test_gaussian_dist_new() {
        let g = GaussianDist::new(0.0, 1.0).unwrap();
        assert!((g.mean - 0.0).abs() < TOLERANCE);
        assert!((g.std_dev - 1.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_gaussian_dist_negative_std() {
        let result = GaussianDist::new(0.0, -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_gaussian_pdf() {
        let g = GaussianDist::standard();
        // pdf(0) = 1/√(2π) ≈ 0.3989
        let pdf_at_zero = g.pdf(0.0);
        assert!((pdf_at_zero - 0.3989422804).abs() < 1e-6);

        // pdf(1) = pdf(-1) by symmetry
        assert!((g.pdf(1.0) - g.pdf(-1.0)).abs() < TOLERANCE);
    }

    #[test]
    fn test_gaussian_cdf() {
        let g = GaussianDist::standard();
        // CDF(0) = 0.5
        assert!((g.cdf(0.0) - 0.5).abs() < 0.001);
        // CDF(-∞) → 0, CDF(∞) → 1
        assert!(g.cdf(-10.0) < 0.001);
        assert!(g.cdf(10.0) > 0.999);
    }

    #[test]
    fn test_gaussian_entropy() {
        let g = GaussianDist::standard();
        // H(N(0,1)) = 0.5 * ln(2πe) ≈ 1.4189
        let expected = 0.5 * (2.0 * PI * E).ln();
        assert!((g.entropy() - expected).abs() < TOLERANCE);
    }

    #[test]
    fn test_gaussian_kl_divergence() {
        let p = GaussianDist::standard();
        let q = GaussianDist::new(0.0, 2.0).unwrap();

        // KL(N(0,1) || N(0,4)) = ln(2) + (1)/(8) - 0.5 ≈ 0.318
        let kl = p.kl_divergence(&q).unwrap();
        let expected = 2.0_f64.ln() + 1.0 / 8.0 - 0.5;
        assert!((kl - expected).abs() < TOLERANCE);
    }

    #[test]
    fn test_gaussian_reparameterization() {
        let g = GaussianDist::new(5.0, 2.0).unwrap();

        // ε = 0 → sample = mean
        assert!((g.sample_reparam(0.0) - 5.0).abs() < TOLERANCE);

        // ε = 1 → sample = mean + std_dev
        assert!((g.sample_reparam(1.0) - 7.0).abs() < TOLERANCE);

        // ε = -1 → sample = mean - std_dev
        assert!((g.sample_reparam(-1.0) - 3.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_gaussian_convolution() {
        // N(1, 4) + N(2, 9) = N(3, 13)
        let g1 = GaussianDist::new(1.0, 2.0).unwrap(); // N(1, 4)
        let g2 = GaussianDist::new(2.0, 3.0).unwrap(); // N(2, 9)

        let sum = g1.convolve(&g2);
        assert!((sum.mean - 3.0).abs() < TOLERANCE);
        assert!((sum.variance() - 13.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_kernel_apply() {
        // K: x ↦ N(2x + 1, 0.25)
        let k = GaussianKernel::linear(2.0, 1.0, 0.5);

        let output = k.apply(3.0);
        assert!((output.mean - 7.0).abs() < TOLERANCE);
        assert!((output.std_dev - 0.5).abs() < TOLERANCE);
    }

    #[test]
    fn test_kernel_composition() {
        // K₁: x ↦ N(2x, 1)
        let k1 = GaussianKernel::linear(2.0, 0.0, 1.0);
        // K₂: y ↦ N(3y, 4)
        let k2 = GaussianKernel::linear(3.0, 0.0, 2.0);

        // K₂ ∘ K₁: x ↦ N(6x, 9*1 + 4) = N(6x, 13)
        let composed = k2.compose(&k1);

        assert!((composed.scale - 6.0).abs() < TOLERANCE);
        assert!((composed.bias - 0.0).abs() < TOLERANCE);
        // Variance: 3² * 1² + 2² = 9 + 4 = 13
        assert!((composed.noise_variance() - 13.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_kernel_identity_composition() {
        let k = GaussianKernel::linear(2.0, 3.0, 1.0);
        let id = GaussianKernel::identity();

        // id ∘ k = k
        let left = id.compose(&k);
        assert!((left.scale - k.scale).abs() < TOLERANCE);
        assert!((left.bias - k.bias).abs() < TOLERANCE);
        assert!((left.noise_std - k.noise_std).abs() < TOLERANCE);

        // k ∘ id = k
        let right = k.compose(&id);
        assert!((right.scale - k.scale).abs() < TOLERANCE);
        assert!((right.bias - k.bias).abs() < TOLERANCE);
        assert!((right.noise_std - k.noise_std).abs() < TOLERANCE);
    }

    #[test]
    fn test_kernel_associativity() {
        let k1 = GaussianKernel::linear(2.0, 1.0, 0.5);
        let k2 = GaussianKernel::linear(0.5, 2.0, 1.0);
        let k3 = GaussianKernel::linear(3.0, -1.0, 0.25);

        // (k3 ∘ k2) ∘ k1 = k3 ∘ (k2 ∘ k1)
        let left = k3.compose(&k2).compose(&k1);
        let right = k3.compose(&k2.compose(&k1));

        assert!((left.scale - right.scale).abs() < TOLERANCE);
        assert!((left.bias - right.bias).abs() < TOLERANCE);
        assert!((left.noise_variance() - right.noise_variance()).abs() < TOLERANCE);
    }

    #[test]
    fn test_multivariate_gaussian() {
        let mean = vec![1.0, 2.0];
        let cov = vec![1.0, 0.5, 0.5, 2.0]; // Symmetric 2x2

        let g = MultivariateGaussian::new(mean, cov).unwrap();
        assert_eq!(g.dim, 2);
        assert!((g.mean[0] - 1.0).abs() < TOLERANCE);
        assert!((g.cov(0, 1) - 0.5).abs() < TOLERANCE);
    }

    #[test]
    fn test_multivariate_spherical() {
        let g = MultivariateGaussian::spherical(vec![0.0, 0.0, 0.0], 2.0).unwrap();
        assert_eq!(g.dim, 3);
        assert!((g.cov(0, 0) - 2.0).abs() < TOLERANCE);
        assert!((g.cov(0, 1) - 0.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_multivariate_kernel_identity() {
        let id = MultivariateGaussianKernel::identity(2);
        let x = vec![3.0, 4.0];

        let output = id.apply(&x).unwrap();
        assert!((output.mean[0] - 3.0).abs() < TOLERANCE);
        assert!((output.mean[1] - 4.0).abs() < TOLERANCE);
        assert!((output.cov(0, 0) - 0.0).abs() < TOLERANCE); // No noise
    }

    #[test]
    fn test_multivariate_kernel_composition() {
        // K₁: ℝ² → ℝ² with A = I, b = 0, Σ = I
        let k1 = MultivariateGaussianKernel::new(
            vec![1.0, 0.0, 0.0, 1.0], // Identity transform
            vec![0.0, 0.0],           // Zero bias
            vec![1.0, 0.0, 0.0, 1.0], // Unit noise covariance
            2,
            2,
        )
        .unwrap();

        // K₂: ℝ² → ℝ² with A = 2I, b = (1,1), Σ = 0.5I
        let k2 = MultivariateGaussianKernel::new(
            vec![2.0, 0.0, 0.0, 2.0], // Scale by 2
            vec![1.0, 1.0],           // Bias
            vec![0.5, 0.0, 0.0, 0.5], // Noise
            2,
            2,
        )
        .unwrap();

        // K₂ ∘ K₁: A = 2I, b = (1,1), Σ = 4I + 0.5I = 4.5I
        let composed = k2.compose(&k1).unwrap();

        assert!((composed.transform[0] - 2.0).abs() < TOLERANCE);
        assert!((composed.bias[0] - 1.0).abs() < TOLERANCE);
        // Noise: 2² * 1 + 0.5 = 4.5
        assert!((composed.noise_cov[0] - 4.5).abs() < TOLERANCE);
    }

    #[test]
    fn test_gauss_category_identity() {
        let id = Gauss::id(2);
        assert_eq!(id.input_dim, 2);
        assert_eq!(id.output_dim, 2);
    }
}
