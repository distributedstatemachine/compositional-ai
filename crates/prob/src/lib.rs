//! # Prob - Stochastic Maps via Category Theory (Sessions 11-13)
//!
//! This crate implements probability distributions and Markov kernels as categorical
//! structures, following the treatment in Tobias Fritz's "Markov categories" and
//! Baez/Fong's work on stochastic matrices as morphisms.
//!
//! ## Core Concepts
//!
//! - **Probability is compositional**: Stochastic maps compose just like functions
//! - **Row-stochastic matrices are morphisms**: `K[i,j]` = P(output=j | input=i)
//! - **Composition is marginalization**: `(K;L)[i,k]` = Σⱼ `K[i,j]` · `L[j,k]`
//! - **The category FinStoch**: Objects are finite sets, morphisms are Markov kernels
//! - **Bayesian networks**: Compositions of conditionals (kernels) over a DAG
//! - **Conditioning**: Bayes' rule requires renormalization (not plain composition)
//!
//! ## Example: Weather Markov Chain
//!
//! ```rust
//! use compositional_prob::{Dist, Kernel};
//!
//! // Weather: Sunny=0, Rainy=1
//! let weather = Kernel::new(vec![
//!     vec![0.8, 0.2],  // Sunny → 80% Sunny, 20% Rainy
//!     vec![0.4, 0.6],  // Rainy → 40% Sunny, 60% Rainy
//! ]).unwrap();
//!
//! // Start sunny
//! let today = Dist::point(2, 0);
//!
//! // Tomorrow's forecast
//! let tomorrow = weather.apply(&today).unwrap();
//! assert!((tomorrow.p[0] - 0.8).abs() < 1e-6);  // 80% sunny
//!
//! // Two days later
//! let two_days = weather.compose(&weather).unwrap();
//! let day_after = two_days.apply(&today).unwrap();
//! ```

mod bayesnet;
mod dist;
mod error;
mod inference;
mod kernel;

pub use bayesnet::{
    naive_bayes, sprinkler_network, BayesNet, Factor, JointDist, VariableElimination,
};
pub use dist::Dist;
pub use error::ProbError;
pub use inference::{
    bayes_update, conditional_mutual_information, evidence_probability, importance_sample,
    likelihood_ratio, likelihood_weighting, likelihood_weighting_with_diagnostics,
    mutual_information, odds_to_prob, odds_update, prob_to_odds, sequential_update, ExactInference,
    ImportanceSampleResult, LikelihoodWeightingResult,
};
pub use kernel::Kernel;

/// Tolerance for probability comparisons.
pub const PROB_TOLERANCE: f32 = 1e-6;

/// The category of finite stochastic maps (FinStoch).
///
/// - Objects: Natural numbers (cardinalities of finite sets)
/// - Morphisms: Row-stochastic matrices (Markov kernels)
/// - Composition: Matrix multiplication
/// - Identity: The identity matrix
pub struct FinStoch;

impl FinStoch {
    /// Identity morphism for a finite set of size n.
    pub fn id(n: usize) -> Kernel {
        Kernel::identity(n)
    }

    /// Compose two morphisms (Markov kernels).
    pub fn compose(f: &Kernel, g: &Kernel) -> Result<Kernel, ProbError> {
        f.compose(g)
    }

    /// The unique morphism from the terminal object (1) to any object.
    /// This represents a probability distribution as a morphism 1 → n.
    pub fn distribution(dist: &Dist) -> Kernel {
        Kernel::constant(1, dist)
    }

    /// The unique morphism to the terminal object (1).
    /// This marginalizes out all information.
    pub fn discard(n: usize) -> Kernel {
        let one = Dist::point(1, 0);
        Kernel::constant(n, &one)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finstoch_identity_composition() {
        let k = Kernel::new(vec![vec![0.3, 0.7], vec![0.5, 0.5]]).unwrap();

        // id ; k = k
        let id_left = FinStoch::id(2);
        let composed_left = FinStoch::compose(&id_left, &k).unwrap();
        assert!(kernel_approx_eq(&composed_left, &k));

        // k ; id = k
        let id_right = FinStoch::id(2);
        let composed_right = FinStoch::compose(&k, &id_right).unwrap();
        assert!(kernel_approx_eq(&composed_right, &k));
    }

    #[test]
    fn test_finstoch_associativity() {
        let f = Kernel::new(vec![vec![0.6, 0.4], vec![0.3, 0.7]]).unwrap();

        let g = Kernel::new(vec![vec![0.5, 0.5], vec![0.2, 0.8]]).unwrap();

        let h = Kernel::new(vec![vec![0.9, 0.1], vec![0.1, 0.9]]).unwrap();

        // (f ; g) ; h = f ; (g ; h)
        let fg = FinStoch::compose(&f, &g).unwrap();
        let fgh_left = FinStoch::compose(&fg, &h).unwrap();

        let gh = FinStoch::compose(&g, &h).unwrap();
        let fgh_right = FinStoch::compose(&f, &gh).unwrap();

        assert!(kernel_approx_eq(&fgh_left, &fgh_right));
    }

    #[test]
    fn test_distribution_as_morphism() {
        let p = Dist::new(vec![0.3, 0.7]).unwrap();
        let k = FinStoch::distribution(&p);

        assert_eq!(k.n_inputs, 1);
        assert_eq!(k.n_outputs, 2);

        // Applying to the unique element gives back p
        let one = Dist::point(1, 0);
        let result = k.apply(&one).unwrap();
        assert!(dist_approx_eq(&result, &p));
    }

    #[test]
    fn test_discard() {
        let d = FinStoch::discard(3);
        assert_eq!(d.n_inputs, 3);
        assert_eq!(d.n_outputs, 1);

        // Any distribution becomes the point mass
        let p = Dist::new(vec![0.2, 0.5, 0.3]).unwrap();
        let result = d.apply(&p).unwrap();
        assert_eq!(result.p.len(), 1);
        assert!((result.p[0] - 1.0).abs() < PROB_TOLERANCE);
    }

    fn kernel_approx_eq(a: &Kernel, b: &Kernel) -> bool {
        if a.n_inputs != b.n_inputs || a.n_outputs != b.n_outputs {
            return false;
        }
        for i in 0..a.n_inputs {
            for j in 0..a.n_outputs {
                if (a.k[i][j] - b.k[i][j]).abs() > PROB_TOLERANCE {
                    return false;
                }
            }
        }
        true
    }

    fn dist_approx_eq(a: &Dist, b: &Dist) -> bool {
        if a.p.len() != b.p.len() {
            return false;
        }
        for i in 0..a.p.len() {
            if (a.p[i] - b.p[i]).abs() > PROB_TOLERANCE {
                return false;
            }
        }
        true
    }
}
