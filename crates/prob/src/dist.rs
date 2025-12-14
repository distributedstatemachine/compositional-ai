//! Probability distributions over finite sets.

use crate::error::ProbError;
use crate::PROB_TOLERANCE;

/// A probability distribution over a finite set {0, 1, ..., n-1}.
///
/// Invariants:
/// - All probabilities are non-negative
/// - Probabilities sum to 1 (within tolerance)
///
/// # Example
///
/// ```rust
/// use compositional_prob::Dist;
///
/// // Fair coin
/// let coin = Dist::uniform(2);
/// assert!((coin.p[0] - 0.5).abs() < 1e-6);
///
/// // Biased die
/// let die = Dist::new(vec![0.1, 0.1, 0.2, 0.2, 0.2, 0.2]).unwrap();
///
/// // Point mass (certain outcome)
/// let certain = Dist::point(3, 1);  // Definitely outcome 1
/// assert_eq!(certain.p[1], 1.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Dist {
    /// Probability vector (sums to 1).
    pub p: Vec<f32>,
}

impl Dist {
    /// Create a new distribution from a probability vector.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The vector is empty
    /// - Any probability is negative
    /// - The probabilities don't sum to 1 (within tolerance)
    pub fn new(p: Vec<f32>) -> Result<Self, ProbError> {
        if p.is_empty() {
            return Err(ProbError::EmptyDistribution);
        }

        // Check for negative probabilities
        if p.iter().any(|&x| x < -PROB_TOLERANCE) {
            return Err(ProbError::NegativeProbability);
        }

        // Check normalization
        let sum: f32 = p.iter().sum();
        if (sum - 1.0).abs() > PROB_TOLERANCE {
            return Err(ProbError::NotNormalized { sum });
        }

        Ok(Self { p })
    }

    /// Create a distribution from unnormalized weights.
    ///
    /// The weights will be normalized to sum to 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// use compositional_prob::Dist;
    ///
    /// let d = Dist::from_weights(vec![1.0, 2.0, 3.0]).unwrap();
    /// // Normalized: [1/6, 2/6, 3/6] = [0.167, 0.333, 0.5]
    /// assert!((d.p[0] - 1.0/6.0).abs() < 1e-6);
    /// ```
    pub fn from_weights(weights: Vec<f32>) -> Result<Self, ProbError> {
        if weights.is_empty() {
            return Err(ProbError::EmptyDistribution);
        }

        if weights.iter().any(|&x| x < 0.0) {
            return Err(ProbError::NegativeProbability);
        }

        let sum: f32 = weights.iter().sum();
        if sum <= 0.0 {
            return Err(ProbError::ZeroWeights);
        }

        let p: Vec<f32> = weights.iter().map(|w| w / sum).collect();
        Ok(Self { p })
    }

    /// Create a uniform distribution over n elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// use compositional_prob::Dist;
    ///
    /// let fair_die = Dist::uniform(6);
    /// assert!((fair_die.p[0] - 1.0/6.0).abs() < 1e-6);
    /// ```
    pub fn uniform(n: usize) -> Self {
        assert!(n > 0, "Cannot create uniform distribution over empty set");
        Self {
            p: vec![1.0 / n as f32; n],
        }
    }

    /// Create a point mass (Dirac delta) at index i.
    ///
    /// P(i) = 1, P(j) = 0 for j ≠ i
    ///
    /// # Example
    ///
    /// ```rust
    /// use compositional_prob::Dist;
    ///
    /// let d = Dist::point(4, 2);  // Certainly outcome 2
    /// assert_eq!(d.p[2], 1.0);
    /// assert_eq!(d.p[0], 0.0);
    /// ```
    pub fn point(n: usize, i: usize) -> Self {
        assert!(i < n, "Index {} out of bounds for size {}", i, n);
        let mut p = vec![0.0; n];
        p[i] = 1.0;
        Self { p }
    }

    /// The support size (number of outcomes with non-zero probability).
    pub fn support_size(&self) -> usize {
        self.p.iter().filter(|&&x| x > PROB_TOLERANCE).count()
    }

    /// The support (indices with non-zero probability).
    pub fn support(&self) -> Vec<usize> {
        self.p
            .iter()
            .enumerate()
            .filter(|(_, &x)| x > PROB_TOLERANCE)
            .map(|(i, _)| i)
            .collect()
    }

    /// The number of outcomes in the sample space.
    pub fn len(&self) -> usize {
        self.p.len()
    }

    /// Check if the distribution is over an empty set (always false for valid Dist).
    pub fn is_empty(&self) -> bool {
        self.p.is_empty()
    }

    /// Get the probability of outcome i.
    pub fn prob(&self, i: usize) -> Result<f32, ProbError> {
        if i >= self.p.len() {
            return Err(ProbError::IndexOutOfBounds {
                index: i,
                size: self.p.len(),
            });
        }
        Ok(self.p[i])
    }

    /// Shannon entropy: `H(p) = -Σ p[i] * log(p[i])`
    ///
    /// Uses natural logarithm. Outcomes with zero probability contribute 0.
    pub fn entropy(&self) -> f32 {
        self.p
            .iter()
            .filter(|&&x| x > PROB_TOLERANCE)
            .map(|&x| -x * x.ln())
            .sum()
    }

    /// Shannon entropy in bits: `H(p) = -Σ p[i] * log2(p[i])`
    pub fn entropy_bits(&self) -> f32 {
        self.entropy() / 2.0_f32.ln()
    }

    /// Mode: the index with highest probability.
    ///
    /// Returns the first index if there are ties.
    pub fn mode(&self) -> usize {
        self.p
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Expected value when outcomes are 0, 1, 2, ..., n-1.
    pub fn mean(&self) -> f32 {
        self.p.iter().enumerate().map(|(i, &p)| i as f32 * p).sum()
    }

    /// Variance when outcomes are 0, 1, 2, ..., n-1.
    pub fn variance(&self) -> f32 {
        let mu = self.mean();
        self.p
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let diff = i as f32 - mu;
                diff * diff * p
            })
            .sum()
    }

    /// Total variation distance between two distributions.
    ///
    /// `TV(p, q) = 0.5 * Σ |p[i] - q[i]|`
    pub fn tv_distance(&self, other: &Dist) -> Result<f32, ProbError> {
        if self.p.len() != other.p.len() {
            return Err(ProbError::ShapeMismatch {
                expected: self.p.len(),
                got: other.p.len(),
            });
        }

        let sum: f32 = self
            .p
            .iter()
            .zip(other.p.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        Ok(sum / 2.0)
    }

    /// KL divergence: `D_KL(self || other) = Σ p[i] * log(p[i] / q[i])`
    ///
    /// Returns infinity if self has support where other has zero probability.
    pub fn kl_divergence(&self, other: &Dist) -> Result<f32, ProbError> {
        if self.p.len() != other.p.len() {
            return Err(ProbError::ShapeMismatch {
                expected: self.p.len(),
                got: other.p.len(),
            });
        }

        let mut kl = 0.0;
        for (p, q) in self.p.iter().zip(other.p.iter()) {
            if *p > PROB_TOLERANCE {
                if *q <= PROB_TOLERANCE {
                    return Ok(f32::INFINITY);
                }
                kl += p * (p / q).ln();
            }
        }
        Ok(kl)
    }

    /// Sample from the distribution using a uniform random value in [0, 1).
    ///
    /// This uses inverse transform sampling.
    pub fn sample(&self, u: f32) -> usize {
        let mut cumsum = 0.0;
        for (i, &p) in self.p.iter().enumerate() {
            cumsum += p;
            if u < cumsum {
                return i;
            }
        }
        // Edge case: u = 1.0 or floating point issues
        self.p.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dist_new_valid() {
        let d = Dist::new(vec![0.3, 0.7]).unwrap();
        assert_eq!(d.p.len(), 2);
        assert!((d.p[0] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_dist_new_not_normalized() {
        let result = Dist::new(vec![0.3, 0.6]);
        assert!(matches!(result, Err(ProbError::NotNormalized { .. })));
    }

    #[test]
    fn test_dist_new_negative() {
        let result = Dist::new(vec![-0.5, 1.5]);
        assert!(matches!(result, Err(ProbError::NegativeProbability)));
    }

    #[test]
    fn test_dist_from_weights() {
        let d = Dist::from_weights(vec![1.0, 2.0, 3.0]).unwrap();
        assert!((d.p[0] - 1.0 / 6.0).abs() < 1e-6);
        assert!((d.p[1] - 2.0 / 6.0).abs() < 1e-6);
        assert!((d.p[2] - 3.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_dist_uniform() {
        let d = Dist::uniform(4);
        for p in &d.p {
            assert!((p - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_dist_point() {
        let d = Dist::point(5, 2);
        assert_eq!(d.p[2], 1.0);
        assert_eq!(d.p[0], 0.0);
        assert_eq!(d.p[4], 0.0);
    }

    #[test]
    fn test_entropy_uniform() {
        let d = Dist::uniform(4);
        // H(uniform) = log(n)
        let expected = 4.0_f32.ln();
        assert!((d.entropy() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_entropy_point_mass() {
        let d = Dist::point(4, 0);
        // H(delta) = 0
        assert!((d.entropy() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_tv_distance() {
        let p = Dist::new(vec![0.5, 0.5]).unwrap();
        let q = Dist::new(vec![1.0, 0.0]).unwrap();
        // TV = 0.5 * (|0.5-1| + |0.5-0|) = 0.5 * 1 = 0.5
        let tv = p.tv_distance(&q).unwrap();
        assert!((tv - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_kl_divergence() {
        let p = Dist::new(vec![0.5, 0.5]).unwrap();
        let q = Dist::uniform(2);
        // KL(p || q) = 0 when p = q
        let kl = p.kl_divergence(&q).unwrap();
        assert!((kl - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_sample() {
        let d = Dist::new(vec![0.3, 0.7]).unwrap();
        assert_eq!(d.sample(0.0), 0);
        assert_eq!(d.sample(0.29), 0);
        assert_eq!(d.sample(0.31), 1);
        assert_eq!(d.sample(0.99), 1);
    }

    #[test]
    fn test_support() {
        let d = Dist::new(vec![0.0, 0.5, 0.0, 0.5]).unwrap();
        assert_eq!(d.support(), vec![1, 3]);
        assert_eq!(d.support_size(), 2);
    }

    #[test]
    fn test_mean_variance() {
        // Bernoulli(0.5): outcomes 0 and 1 with equal prob
        let d = Dist::new(vec![0.5, 0.5]).unwrap();
        assert!((d.mean() - 0.5).abs() < 1e-6);
        assert!((d.variance() - 0.25).abs() < 1e-6);
    }
}
