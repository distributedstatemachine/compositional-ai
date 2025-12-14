//! Inference and conditioning for Bayesian reasoning.
//!
//! This module provides tools for:
//! - Bayes' rule updates
//! - Sequential evidence updates
//! - Exact inference via enumeration
//!
//! Key insight: Conditioning is NOT plain composition. It requires renormalization
//! because observing evidence restricts the probability space.

use crate::bayesnet::BayesNet;
use crate::dist::Dist;
use crate::error::ProbError;
use crate::kernel::Kernel;
use std::collections::HashMap;

/// Compute posterior P(H|E=e) using Bayes' rule.
///
/// Given:
/// - prior: P(H) - prior distribution over hypothesis
/// - likelihood: P(E|H) - likelihood as a kernel H → E
/// - evidence_value: the observed value e
///
/// Returns: P(H|E=e) = P(E=e|H) · P(H) / P(E=e)
///
/// # Example
///
/// ```rust
/// use compositional_prob::{Dist, Kernel, bayes_update};
///
/// // Disease prevalence: 1% have disease
/// let prior = Dist::new(vec![0.99, 0.01]).unwrap();
///
/// // Test accuracy: P(Test | Disease)
/// let test = Kernel::new(vec![
///     vec![0.95, 0.05],  // Healthy: 5% false positive
///     vec![0.10, 0.90],  // Diseased: 90% true positive
/// ]).unwrap();
///
/// // Patient tests positive (evidence_value = 1)
/// let posterior = bayes_update(&prior, &test, 1);
///
/// // Despite positive test, still unlikely to have disease!
/// // P(D|+) ≈ 0.154 due to low base rate
/// assert!(posterior.p[1] < 0.2);
/// ```
pub fn bayes_update(prior: &Dist, likelihood: &Kernel, evidence_value: usize) -> Dist {
    assert!(
        evidence_value < likelihood.n_outputs,
        "Evidence value {} out of bounds for likelihood with {} outputs",
        evidence_value,
        likelihood.n_outputs
    );
    assert_eq!(
        prior.p.len(),
        likelihood.n_inputs,
        "Prior size {} doesn't match likelihood inputs {}",
        prior.p.len(),
        likelihood.n_inputs
    );

    // Compute unnormalized posterior: P(H=h) · P(E=e|H=h)
    let unnormalized: Vec<f32> = prior
        .p
        .iter()
        .enumerate()
        .map(|(h, &prior_h)| prior_h * likelihood.k[h][evidence_value])
        .collect();

    // Normalize to get P(H|E=e)
    Dist::from_weights(unnormalized).expect("Evidence has zero probability")
}

/// Compute the evidence probability P(E=e) (the normalizing constant).
///
/// P(E=e) = Σₕ P(H=h) · P(E=e|H=h)
pub fn evidence_probability(prior: &Dist, likelihood: &Kernel, evidence_value: usize) -> f32 {
    prior
        .p
        .iter()
        .enumerate()
        .map(|(h, &prior_h)| prior_h * likelihood.k[h][evidence_value])
        .sum()
}

/// Perform sequential Bayesian updates with multiple pieces of evidence.
///
/// Each update uses the posterior from the previous step as the new prior.
///
/// # Example
///
/// ```rust
/// use compositional_prob::{Dist, Kernel, sequential_update};
///
/// let prior = Dist::new(vec![0.5, 0.5]).unwrap();
///
/// // Two different tests
/// let test1 = Kernel::new(vec![
///     vec![0.9, 0.1],
///     vec![0.3, 0.7],
/// ]).unwrap();
///
/// let test2 = Kernel::new(vec![
///     vec![0.8, 0.2],
///     vec![0.2, 0.8],
/// ]).unwrap();
///
/// // Both tests positive
/// let observations = vec![(&test1, 1), (&test2, 1)];
/// let posterior = sequential_update(&prior, &observations);
/// ```
pub fn sequential_update(prior: &Dist, observations: &[(&Kernel, usize)]) -> Dist {
    let mut current = prior.clone();

    for (likelihood, evidence_value) in observations {
        current = bayes_update(&current, likelihood, *evidence_value);
    }

    current
}

/// Compute the likelihood ratio P(E=e|H=1) / P(E=e|H=0) for binary hypothesis.
///
/// The likelihood ratio tells us how much more likely the evidence is
/// under H=1 vs H=0. Used in sequential hypothesis testing.
pub fn likelihood_ratio(likelihood: &Kernel, evidence_value: usize) -> f32 {
    assert_eq!(
        likelihood.n_inputs, 2,
        "Likelihood ratio requires binary hypothesis"
    );
    let p_e_given_h0 = likelihood.k[0][evidence_value];
    let p_e_given_h1 = likelihood.k[1][evidence_value];

    if p_e_given_h0 == 0.0 {
        f32::INFINITY
    } else {
        p_e_given_h1 / p_e_given_h0
    }
}

/// Convert between odds and probability.
///
/// odds = p / (1-p)
/// p = odds / (1 + odds)
pub fn prob_to_odds(p: f32) -> f32 {
    if p >= 1.0 {
        f32::INFINITY
    } else {
        p / (1.0 - p)
    }
}

/// Convert odds back to probability.
pub fn odds_to_prob(odds: f32) -> f32 {
    if odds.is_infinite() {
        1.0
    } else {
        odds / (1.0 + odds)
    }
}

/// Update odds using likelihood ratio (for binary hypothesis).
///
/// posterior_odds = prior_odds × likelihood_ratio
///
/// This is equivalent to Bayes' rule but often more convenient for
/// sequential updates.
pub fn odds_update(prior_odds: f32, likelihood_ratio: f32) -> f32 {
    prior_odds * likelihood_ratio
}

/// Exact inference engine for Bayesian networks.
///
/// Uses enumeration over the full joint distribution.
/// Suitable for small networks (< 20 binary variables).
pub struct ExactInference<'a> {
    net: &'a BayesNet,
    joint: Option<crate::bayesnet::JointDist>,
}

impl<'a> ExactInference<'a> {
    /// Create a new exact inference engine.
    pub fn new(net: &'a BayesNet) -> Self {
        Self { net, joint: None }
    }

    /// Compute and cache the full joint distribution.
    pub fn compute_joint(&mut self) {
        if self.joint.is_none() {
            self.joint = Some(self.net.full_joint());
        }
    }

    /// Get the cached joint, computing if necessary.
    fn get_joint(&mut self) -> &crate::bayesnet::JointDist {
        self.compute_joint();
        self.joint.as_ref().unwrap()
    }

    /// Compute marginal P(variable).
    pub fn marginal(&mut self, variable: usize) -> Result<Dist, ProbError> {
        let joint = self.get_joint();
        joint.marginalize_to(&[variable])
    }

    /// Compute conditional P(query | evidence).
    pub fn query(
        &mut self,
        query_var: usize,
        evidence: &HashMap<usize, usize>,
    ) -> Result<Dist, ProbError> {
        let joint = self.get_joint();
        joint.condition_on(query_var, evidence)
    }

    /// Compute joint marginal P(variables).
    pub fn joint_marginal(&mut self, variables: &[usize]) -> Result<Dist, ProbError> {
        let joint = self.get_joint();
        joint.marginalize_to(variables)
    }

    /// Compute P(evidence) - the probability of the evidence.
    pub fn evidence_prob(&mut self, evidence: &HashMap<usize, usize>) -> f32 {
        let joint = self.get_joint();
        let mut prob = 0.0;

        for (idx, &p) in joint.probs.iter().enumerate() {
            let assignment = joint.decode(idx);
            let matches = evidence.iter().all(|(&var, &val)| assignment[var] == val);
            if matches {
                prob += p;
            }
        }

        prob
    }

    /// Check if two variables are conditionally independent given a set.
    ///
    /// X ⊥ Y | Z iff P(X,Y|Z) = P(X|Z) · P(Y|Z)
    pub fn conditionally_independent(
        &mut self,
        x: usize,
        y: usize,
        z: &[usize],
        tolerance: f32,
    ) -> bool {
        // For each value of Z, check if X and Y are independent
        let z_states: Vec<usize> = z.iter().map(|&v| self.net.var_states[v]).collect();
        let total_z: usize = if z_states.is_empty() {
            1
        } else {
            z_states.iter().product()
        };

        for z_idx in 0..total_z {
            // Decode Z values
            let mut evidence: HashMap<usize, usize> = HashMap::new();
            let mut idx = z_idx;
            for (i, &zv) in z.iter().enumerate() {
                evidence.insert(zv, idx % z_states[i]);
                idx /= z_states[i];
            }

            // Get P(X|Z), P(Y|Z), P(X,Y|Z)
            let p_x_given_z = match self.query(x, &evidence) {
                Ok(d) => d,
                Err(_) => continue, // Z has zero probability
            };
            let p_y_given_z = match self.query(y, &evidence) {
                Ok(d) => d,
                Err(_) => continue,
            };

            // Check P(X=x, Y=y | Z) ≈ P(X=x|Z) · P(Y=y|Z) for all x, y
            for x_val in 0..self.net.var_states[x] {
                for y_val in 0..self.net.var_states[y] {
                    let mut xy_evidence = evidence.clone();
                    xy_evidence.insert(x, x_val);
                    xy_evidence.insert(y, y_val);

                    // P(X=x, Y=y | Z) via joint query
                    let p_xy_z = self.evidence_prob(&xy_evidence);
                    let p_z = self.evidence_prob(&evidence);

                    if p_z < tolerance {
                        continue;
                    }

                    let p_xy_given_z = p_xy_z / p_z;
                    let p_x_times_y = p_x_given_z.p[x_val] * p_y_given_z.p[y_val];

                    if (p_xy_given_z - p_x_times_y).abs() > tolerance {
                        return false;
                    }
                }
            }
        }

        true
    }
}

/// Compute the mutual information I(X; Y) between two variables.
///
/// I(X; Y) = Σₓ,ᵧ P(x,y) · log(P(x,y) / (P(x)·P(y)))
pub fn mutual_information(net: &BayesNet, x: usize, y: usize) -> f32 {
    let joint = net.full_joint();

    let p_x = net.marginal(x).unwrap();
    let p_y = net.marginal(y).unwrap();

    // Compute joint marginal P(X, Y) by summing over other variables
    let n_x = net.var_states[x];
    let n_y = net.var_states[y];
    let mut p_xy = vec![vec![0.0; n_y]; n_x];

    for (idx, &p) in joint.probs.iter().enumerate() {
        let assignment = joint.decode(idx);
        let x_val = assignment[x];
        let y_val = assignment[y];
        p_xy[x_val][y_val] += p;
    }

    // Compute mutual information
    let mut mi = 0.0;

    for (x_val, row) in p_xy.iter().enumerate() {
        for (y_val, &p_joint) in row.iter().enumerate() {
            if p_joint > 1e-10 {
                let p_x_val = p_x.p[x_val];
                let p_y_val = p_y.p[y_val];

                if p_x_val > 1e-10 && p_y_val > 1e-10 {
                    mi += p_joint * (p_joint / (p_x_val * p_y_val)).ln();
                }
            }
        }
    }

    mi
}

/// Compute conditional mutual information I(X; Y | Z).
///
/// I(X; Y | Z) = Σ_z P(z) · I(X; Y | Z=z)
pub fn conditional_mutual_information(net: &BayesNet, x: usize, y: usize, z: usize) -> f32 {
    let p_z = net.marginal(z).unwrap();
    let joint = net.full_joint();

    let mut cmi = 0.0;

    for z_val in 0..net.var_states[z] {
        if p_z.p[z_val] < 1e-10 {
            continue;
        }

        // Compute I(X; Y | Z=z_val)
        let mut evidence = HashMap::new();
        evidence.insert(z, z_val);

        let p_x_given_z = match net.query(x, &evidence) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let p_y_given_z = match net.query(y, &evidence) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let mut mi_given_z = 0.0;

        for (idx, &p) in joint.probs.iter().enumerate() {
            let assignment = joint.decode(idx);
            if assignment[z] != z_val {
                continue;
            }

            let x_val = assignment[x];
            let y_val = assignment[y];

            // P(x, y | z) = P(x, y, z) / P(z)
            let p_xy_given_z = p / p_z.p[z_val];

            if p_xy_given_z > 1e-10 {
                let p_x_z = p_x_given_z.p[x_val];
                let p_y_z = p_y_given_z.p[y_val];

                if p_x_z > 1e-10 && p_y_z > 1e-10 {
                    mi_given_z += p_xy_given_z * (p_xy_given_z / (p_x_z * p_y_z)).ln();
                }
            }
        }

        cmi += p_z.p[z_val] * mi_given_z;
    }

    cmi
}

// =============================================================================
// Importance Sampling
// =============================================================================

/// Result of importance sampling estimation.
#[derive(Debug, Clone)]
pub struct ImportanceSampleResult {
    /// Estimated expectation E[f(X)] under target distribution.
    pub estimate: f32,
    /// Effective sample size (measure of sample quality).
    pub effective_sample_size: f32,
    /// Number of samples used.
    pub n_samples: usize,
    /// Variance of the importance weights.
    pub weight_variance: f32,
}

/// Importance sampling for estimating expectations under a target distribution.
///
/// When exact inference is intractable, importance sampling provides an
/// unbiased estimate by sampling from a proposal distribution and reweighting.
///
/// # The Idea
///
/// To estimate E_p[f(X)] where p is hard to sample from:
/// 1. Sample from an easier proposal distribution q(x)
/// 2. Reweight samples by w(x) = p(x) / q(x)
/// 3. Estimate = Σᵢ wᵢ · f(xᵢ) / Σᵢ wᵢ  (self-normalized)
///
/// # Example
///
/// ```rust
/// use compositional_prob::{Dist, importance_sample};
///
/// // Target: want to estimate mean of this distribution
/// let target = Dist::new(vec![0.1, 0.2, 0.3, 0.4]).unwrap();
///
/// // Proposal: uniform (easy to sample from)
/// let proposal = Dist::uniform(4);
///
/// // Function: identity (to estimate the mean)
/// let f = |x: usize| x as f32;
///
/// let result = importance_sample(&target, &proposal, f, 1000, Some(42));
///
/// // True mean = 0*0.1 + 1*0.2 + 2*0.3 + 3*0.4 = 2.0
/// assert!((result.estimate - 2.0).abs() < 0.2);
/// ```
pub fn importance_sample<F>(
    target: &Dist,
    proposal: &Dist,
    f: F,
    n_samples: usize,
    seed: Option<u64>,
) -> ImportanceSampleResult
where
    F: Fn(usize) -> f32,
{
    assert_eq!(
        target.p.len(),
        proposal.p.len(),
        "Target and proposal must have same support"
    );

    // Simple LCG for reproducibility (avoid external deps)
    let mut rng_state = seed.unwrap_or(12345);
    let mut next_random = || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng_state >> 33) as f32 / (1u64 << 31) as f32
    };

    // Sample from proposal and compute weights
    let mut weights = Vec::with_capacity(n_samples);
    let mut weighted_values = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        // Sample from proposal using inverse CDF
        let u = next_random();
        let mut cumsum = 0.0;
        let mut sample = 0;
        for (i, &p) in proposal.p.iter().enumerate() {
            cumsum += p;
            if u <= cumsum {
                sample = i;
                break;
            }
        }

        // Importance weight: w = p(x) / q(x)
        let q_x = proposal.p[sample];
        let p_x = target.p[sample];

        if q_x > 1e-10 {
            let weight = p_x / q_x;
            weights.push(weight);
            weighted_values.push(weight * f(sample));
        }
    }

    // Self-normalized importance sampling
    let sum_weights: f32 = weights.iter().sum();
    let sum_weighted_values: f32 = weighted_values.iter().sum();

    let estimate = if sum_weights > 1e-10 {
        sum_weighted_values / sum_weights
    } else {
        0.0
    };

    // Compute effective sample size: ESS = (Σwᵢ)² / Σwᵢ²
    let sum_weights_sq: f32 = weights.iter().map(|w| w * w).sum();
    let effective_sample_size = if sum_weights_sq > 1e-10 {
        (sum_weights * sum_weights) / sum_weights_sq
    } else {
        0.0
    };

    // Weight variance
    let mean_weight = sum_weights / weights.len() as f32;
    let weight_variance: f32 = weights
        .iter()
        .map(|w| (w - mean_weight).powi(2))
        .sum::<f32>()
        / weights.len() as f32;

    ImportanceSampleResult {
        estimate,
        effective_sample_size,
        n_samples,
        weight_variance,
    }
}

/// Estimate P(query | evidence) using likelihood weighting.
///
/// Likelihood weighting is importance sampling specialized for Bayesian networks.
/// We sample from the prior (ignoring evidence) and weight by evidence likelihood.
///
/// # Algorithm
///
/// 1. Sample all non-evidence variables from their conditionals (forward sampling)
/// 2. Set evidence variables to their observed values
/// 3. Weight = product of P(evidence_var = observed_val | parents)
/// 4. Estimate posterior as weighted average
///
/// # Example
///
/// ```rust
/// use compositional_prob::{sprinkler_network, likelihood_weighting};
/// use std::collections::HashMap;
///
/// let net = sprinkler_network();
///
/// // Query P(Rain | WetGrass=true)
/// let mut evidence = HashMap::new();
/// evidence.insert(3, 1);  // WetGrass = true
///
/// let result = likelihood_weighting(&net, 2, &evidence, 1000, Some(42));
///
/// // Rain is more likely given wet grass
/// assert!(result.p[1] > 0.5);
/// ```
pub fn likelihood_weighting(
    net: &BayesNet,
    query_var: usize,
    evidence: &HashMap<usize, usize>,
    n_samples: usize,
    seed: Option<u64>,
) -> Dist {
    let n_query_states = net.var_states[query_var];
    let mut weighted_counts = vec![0.0; n_query_states];
    let mut total_weight = 0.0;

    // Simple LCG for reproducibility
    let mut rng_state = seed.unwrap_or(12345);
    let mut next_random = || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng_state >> 33) as f32 / (1u64 << 31) as f32
    };

    // Build topological order (simple version: assume factors are in order)
    let var_order: Vec<usize> = (0..net.n_vars).collect();

    for _ in 0..n_samples {
        let mut assignment = vec![0; net.n_vars];
        let mut weight = 1.0_f32;

        for &var in &var_order {
            // Find the factor for this variable
            let factor = net.factors.iter().find(|f| f.variable == var);

            if let Some(evidence_val) = evidence.get(&var) {
                // Evidence variable: set to observed value and accumulate weight
                assignment[var] = *evidence_val;

                if let Some(f) = factor {
                    let parent_vals: Vec<usize> =
                        f.parents.iter().map(|&p| assignment[p]).collect();
                    weight *= f.prob(*evidence_val, &parent_vals);
                }
            } else {
                // Non-evidence variable: sample from conditional
                if let Some(f) = factor {
                    let parent_vals: Vec<usize> =
                        f.parents.iter().map(|&p| assignment[p]).collect();
                    let parent_idx = f.encode_parents(&parent_vals);

                    // Sample from P(var | parents)
                    let u = next_random();
                    let mut cumsum = 0.0;
                    for state in 0..f.n_states {
                        cumsum += f.cpt.k[parent_idx][state];
                        if u <= cumsum {
                            assignment[var] = state;
                            break;
                        }
                    }
                } else {
                    // No factor found, sample uniformly
                    assignment[var] = (next_random() * net.var_states[var] as f32) as usize;
                }
            }
        }

        // Accumulate weighted counts
        if weight > 1e-10 {
            weighted_counts[assignment[query_var]] += weight;
            total_weight += weight;
        }
    }

    // Normalize - use from_weights to handle floating point drift
    if total_weight > 1e-10 {
        // from_weights handles renormalization properly
        Dist::from_weights(weighted_counts).unwrap_or_else(|_| Dist::uniform(n_query_states))
    } else {
        // Fallback to uniform if all weights are zero
        Dist::uniform(n_query_states)
    }
}

/// Result of likelihood weighting with diagnostics.
#[derive(Debug, Clone)]
pub struct LikelihoodWeightingResult {
    /// Estimated posterior distribution.
    pub posterior: Dist,
    /// Effective sample size.
    pub effective_sample_size: f32,
    /// Number of samples used.
    pub n_samples: usize,
}

/// Likelihood weighting with diagnostics.
///
/// Same as `likelihood_weighting` but returns additional diagnostics
/// about the quality of the estimate.
pub fn likelihood_weighting_with_diagnostics(
    net: &BayesNet,
    query_var: usize,
    evidence: &HashMap<usize, usize>,
    n_samples: usize,
    seed: Option<u64>,
) -> LikelihoodWeightingResult {
    let n_query_states = net.var_states[query_var];
    let mut weighted_counts = vec![0.0; n_query_states];
    let mut weights = Vec::with_capacity(n_samples);

    // Simple LCG for reproducibility
    let mut rng_state = seed.unwrap_or(12345);
    let mut next_random = || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng_state >> 33) as f32 / (1u64 << 31) as f32
    };

    let var_order: Vec<usize> = (0..net.n_vars).collect();

    for _ in 0..n_samples {
        let mut assignment = vec![0; net.n_vars];
        let mut weight = 1.0_f32;

        for &var in &var_order {
            let factor = net.factors.iter().find(|f| f.variable == var);

            if let Some(evidence_val) = evidence.get(&var) {
                assignment[var] = *evidence_val;
                if let Some(f) = factor {
                    let parent_vals: Vec<usize> =
                        f.parents.iter().map(|&p| assignment[p]).collect();
                    weight *= f.prob(*evidence_val, &parent_vals);
                }
            } else if let Some(f) = factor {
                let parent_vals: Vec<usize> = f.parents.iter().map(|&p| assignment[p]).collect();
                let parent_idx = f.encode_parents(&parent_vals);

                let u = next_random();
                let mut cumsum = 0.0;
                for state in 0..f.n_states {
                    cumsum += f.cpt.k[parent_idx][state];
                    if u <= cumsum {
                        assignment[var] = state;
                        break;
                    }
                }
            } else {
                assignment[var] = (next_random() * net.var_states[var] as f32) as usize;
            }
        }

        if weight > 1e-10 {
            weighted_counts[assignment[query_var]] += weight;
            weights.push(weight);
        }
    }

    // Effective sample size (compute before normalizing weighted_counts)
    let total_weight: f32 = weights.iter().sum();
    let sum_weights_sq: f32 = weights.iter().map(|w| w * w).sum();
    let effective_sample_size = if sum_weights_sq > 1e-10 {
        (total_weight * total_weight) / sum_weights_sq
    } else {
        0.0
    };

    // Normalize - use from_weights to handle floating point drift
    let posterior = if total_weight > 1e-10 {
        Dist::from_weights(weighted_counts).unwrap_or_else(|_| Dist::uniform(n_query_states))
    } else {
        Dist::uniform(n_query_states)
    };

    LikelihoodWeightingResult {
        posterior,
        effective_sample_size,
        n_samples,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PROB_TOLERANCE;

    #[test]
    fn test_bayes_update_basic() {
        // Fair prior
        let prior = Dist::new(vec![0.5, 0.5]).unwrap();

        // Biased likelihood
        let likelihood = Kernel::new(vec![
            vec![0.9, 0.1], // H=0: 10% positive
            vec![0.2, 0.8], // H=1: 80% positive
        ])
        .unwrap();

        // Observe positive (E=1)
        let posterior = bayes_update(&prior, &likelihood, 1);

        // P(H=1|E=1) = (0.5 × 0.8) / (0.5 × 0.1 + 0.5 × 0.8) = 0.4 / 0.45 ≈ 0.889
        assert!((posterior.p[1] - 0.8 / 0.9).abs() < 0.01);
    }

    #[test]
    fn test_medical_diagnosis() {
        // 1% disease prevalence
        let prior = Dist::new(vec![0.99, 0.01]).unwrap();

        // Test: 5% false positive, 90% true positive
        let test = Kernel::new(vec![vec![0.95, 0.05], vec![0.10, 0.90]]).unwrap();

        // Positive test
        let posterior = bayes_update(&prior, &test, 1);

        // P(D|+) = (0.01 × 0.90) / (0.99 × 0.05 + 0.01 × 0.90)
        //        = 0.009 / 0.0585 ≈ 0.154
        let expected = 0.009 / 0.0585;
        assert!((posterior.p[1] - expected).abs() < 0.01);
    }

    #[test]
    fn test_evidence_probability() {
        let prior = Dist::new(vec![0.5, 0.5]).unwrap();
        let likelihood = Kernel::new(vec![vec![0.9, 0.1], vec![0.2, 0.8]]).unwrap();

        // P(E=1) = 0.5 × 0.1 + 0.5 × 0.8 = 0.45
        let p_e = evidence_probability(&prior, &likelihood, 1);
        assert!((p_e - 0.45).abs() < PROB_TOLERANCE);
    }

    #[test]
    fn test_sequential_update() {
        let prior = Dist::new(vec![0.5, 0.5]).unwrap();

        let test = Kernel::new(vec![vec![0.8, 0.2], vec![0.3, 0.7]]).unwrap();

        // Two positive tests
        let observations = vec![(&test, 1), (&test, 1)];
        let posterior = sequential_update(&prior, &observations);

        // After two positive tests, should strongly favor H=1
        assert!(posterior.p[1] > 0.8);
    }

    #[test]
    fn test_likelihood_ratio() {
        let likelihood = Kernel::new(vec![
            vec![0.9, 0.1], // H=0
            vec![0.2, 0.8], // H=1
        ])
        .unwrap();

        // LR for E=1: 0.8 / 0.1 = 8
        let lr = likelihood_ratio(&likelihood, 1);
        assert!((lr - 8.0).abs() < PROB_TOLERANCE);

        // LR for E=0: 0.2 / 0.9 ≈ 0.222
        let lr0 = likelihood_ratio(&likelihood, 0);
        assert!((lr0 - 0.2 / 0.9).abs() < 0.01);
    }

    #[test]
    fn test_odds_conversion() {
        // p = 0.75 => odds = 3
        assert!((prob_to_odds(0.75) - 3.0).abs() < PROB_TOLERANCE);

        // odds = 3 => p = 0.75
        assert!((odds_to_prob(3.0) - 0.75).abs() < PROB_TOLERANCE);

        // Round trip
        let p = 0.6;
        assert!((odds_to_prob(prob_to_odds(p)) - p).abs() < PROB_TOLERANCE);
    }

    #[test]
    fn test_exact_inference() {
        use crate::sprinkler_network;

        let net = sprinkler_network();
        let mut engine = ExactInference::new(&net);

        // P(Cloudy) should be [0.5, 0.5]
        let p_cloudy = engine.marginal(0).unwrap();
        assert!((p_cloudy.p[0] - 0.5).abs() < PROB_TOLERANCE);

        // P(Rain | Cloudy=True) should be [0.2, 0.8]
        let mut evidence = HashMap::new();
        evidence.insert(0, 1);
        let p_rain = engine.query(2, &evidence).unwrap();
        assert!((p_rain.p[0] - 0.2).abs() < 0.01);
        assert!((p_rain.p[1] - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_mutual_information() {
        use crate::sprinkler_network;

        let net = sprinkler_network();

        // Cloudy and Rain should have positive MI (they're dependent)
        let mi_cr = mutual_information(&net, 0, 2);
        assert!(mi_cr > 0.0);

        // MI should be symmetric
        let mi_rc = mutual_information(&net, 2, 0);
        assert!((mi_cr - mi_rc).abs() < PROB_TOLERANCE);
    }

    #[test]
    fn test_importance_sampling_basic() {
        // Target distribution
        let target = Dist::new(vec![0.1, 0.2, 0.3, 0.4]).unwrap();

        // Proposal: uniform
        let proposal = Dist::uniform(4);

        // Estimate mean: E[X] = 0*0.1 + 1*0.2 + 2*0.3 + 3*0.4 = 2.0
        let f = |x: usize| x as f32;
        let result = importance_sample(&target, &proposal, f, 10000, Some(42));

        assert!((result.estimate - 2.0).abs() < 0.1);
        assert!(result.effective_sample_size > 0.0);
    }

    #[test]
    fn test_importance_sampling_same_distribution() {
        // When target = proposal, weights should be uniform
        let dist = Dist::new(vec![0.25, 0.25, 0.25, 0.25]).unwrap();

        let f = |x: usize| x as f32;
        let result = importance_sample(&dist, &dist, f, 1000, Some(42));

        // Mean of uniform over {0,1,2,3} = 1.5
        assert!((result.estimate - 1.5).abs() < 0.2);

        // ESS should be close to n_samples when proposal = target
        assert!(result.effective_sample_size > 800.0);
    }

    #[test]
    fn test_likelihood_weighting() {
        use crate::sprinkler_network;

        let net = sprinkler_network();

        // P(Rain | WetGrass=true)
        let mut evidence = HashMap::new();
        evidence.insert(3, 1); // WetGrass = true

        // Get exact answer
        let exact = net.query(2, &evidence).unwrap();

        // Get approximate answer
        let approx = likelihood_weighting(&net, 2, &evidence, 10000, Some(42));

        // Should be close (within 0.1)
        assert!(
            (approx.p[0] - exact.p[0]).abs() < 0.1,
            "Got p[0]={}, expected {}",
            approx.p[0],
            exact.p[0]
        );
        assert!(
            (approx.p[1] - exact.p[1]).abs() < 0.1,
            "Got p[1]={}, expected {}",
            approx.p[1],
            exact.p[1]
        );
    }

    #[test]
    fn test_likelihood_weighting_with_diagnostics() {
        use crate::sprinkler_network;

        let net = sprinkler_network();

        let mut evidence = HashMap::new();
        evidence.insert(3, 1);

        let result = likelihood_weighting_with_diagnostics(&net, 2, &evidence, 5000, Some(42));

        assert_eq!(result.n_samples, 5000);
        assert!(result.effective_sample_size > 0.0);
        assert!((result.posterior.p[0] + result.posterior.p[1] - 1.0).abs() < PROB_TOLERANCE);
    }
}
