//! Causal inference: interventions and do-calculus.
//!
//! This module provides tools for causal reasoning:
//! - Interventions via graph surgery (do-operator)
//! - Backdoor adjustment formula
//! - Average treatment effects
//! - Causal vs observational queries
//!
//! Key insight: do(X=x) ≠ observing X=x
//! - Observing: Updates beliefs about causes of X
//! - Intervening: Breaks the causal mechanism, isolates X's effect

use crate::bayesnet::BayesNet;
use crate::dist::Dist;
use crate::error::ProbError;
use std::collections::HashMap;

/// Compute P(Y | do(X=x)) - the interventional distribution.
///
/// This answers: "What would the distribution of Y be if we forced X=x?"
///
/// # Algorithm
///
/// 1. Create mutilated graph with intervention do(X=x)
/// 2. Compute marginal P(Y) in the mutilated graph
///
/// # Example
///
/// ```rust
/// use compositional_prob::{sprinkler_network, causal::interventional_query};
///
/// let net = sprinkler_network();
///
/// // P(WetGrass | do(Rain=True))
/// let p_wet_do_rain = interventional_query(&net, 3, 2, 1).unwrap();
/// println!("P(Wet | do(Rain=1)) = [{:.3}, {:.3}]", p_wet_do_rain.p[0], p_wet_do_rain.p[1]);
/// ```
pub fn interventional_query(
    net: &BayesNet,
    query_var: usize,
    intervention_var: usize,
    intervention_value: usize,
) -> Result<Dist, ProbError> {
    let mutilated = net.intervene(intervention_var, intervention_value);
    mutilated.marginal(query_var)
}

/// Compute P(Y=y | do(X=x)) for a specific outcome.
pub fn interventional_prob(
    net: &BayesNet,
    query_var: usize,
    query_value: usize,
    intervention_var: usize,
    intervention_value: usize,
) -> Result<f32, ProbError> {
    let dist = interventional_query(net, query_var, intervention_var, intervention_value)?;
    if query_value >= dist.p.len() {
        return Err(ProbError::IndexOutOfBounds {
            index: query_value,
            size: dist.p.len(),
        });
    }
    Ok(dist.p[query_value])
}

/// Compute the Average Treatment Effect (ATE).
///
/// ATE = E[Y | do(X=1)] - E[Y | do(X=0)]
///
/// For binary treatment X and outcome Y, this measures the causal effect
/// of treatment on the outcome.
///
/// # Example
///
/// ```rust
/// use compositional_prob::{causal::average_treatment_effect, BayesNet, Dist, Kernel};
///
/// // Simple treatment -> outcome model
/// let mut net = BayesNet::new(vec![2, 2]); // Treatment, Outcome
/// net.add_prior(0, &Dist::new(vec![0.5, 0.5]).unwrap()).unwrap();
/// net.add_conditional(1, vec![0], Kernel::new(vec![
///     vec![0.8, 0.2],  // No treatment: 20% positive outcome
///     vec![0.3, 0.7],  // Treatment: 70% positive outcome
/// ]).unwrap()).unwrap();
///
/// let ate = average_treatment_effect(&net, 0, 1).unwrap();
/// assert!((ate - 0.5).abs() < 0.01); // Treatment increases outcome by 50%
/// ```
pub fn average_treatment_effect(
    net: &BayesNet,
    treatment_var: usize,
    outcome_var: usize,
) -> Result<f32, ProbError> {
    // E[Y | do(X=1)]
    let p_y_do_1 = interventional_query(net, outcome_var, treatment_var, 1)?;
    let e_y_do_1 = expected_value(&p_y_do_1);

    // E[Y | do(X=0)]
    let p_y_do_0 = interventional_query(net, outcome_var, treatment_var, 0)?;
    let e_y_do_0 = expected_value(&p_y_do_0);

    Ok(e_y_do_1 - e_y_do_0)
}

/// Compute expected value for a distribution over {0, 1, 2, ...}.
fn expected_value(dist: &Dist) -> f32 {
    dist.p.iter().enumerate().map(|(i, &p)| i as f32 * p).sum()
}

/// Compute P(Y | do(X=x)) using the backdoor adjustment formula.
///
/// When we have a set Z of variables that blocks all backdoor paths from X to Y:
///
/// P(Y | do(X=x)) = Σ_z P(Y | X=x, Z=z) · P(Z=z)
///
/// This allows computing interventional distributions from observational data
/// when the adjustment set is observed.
///
/// # Arguments
///
/// * `net` - The Bayesian network
/// * `x_var` - The intervention variable X
/// * `x_val` - The intervention value x
/// * `y_var` - The query variable Y
/// * `adjustment_set` - Variables Z that block backdoor paths
///
/// # Example
///
/// ```rust
/// use compositional_prob::{causal::backdoor_adjustment, BayesNet, Dist, Kernel};
///
/// // Confounder -> Treatment, Confounder -> Outcome, Treatment -> Outcome
/// let mut net = BayesNet::new(vec![2, 2, 2]); // Confounder, Treatment, Outcome
/// net.add_prior(0, &Dist::new(vec![0.5, 0.5]).unwrap()).unwrap();
/// net.add_conditional(1, vec![0], Kernel::new(vec![
///     vec![0.8, 0.2],  // Confounder=0: 20% treated
///     vec![0.2, 0.8],  // Confounder=1: 80% treated
/// ]).unwrap()).unwrap();
/// net.add_conditional(2, vec![0, 1], Kernel::new(vec![
///     vec![0.9, 0.1],  // C=0, T=0: 10% outcome
///     vec![0.7, 0.3],  // C=0, T=1: 30% outcome
///     vec![0.6, 0.4],  // C=1, T=0: 40% outcome
///     vec![0.3, 0.7],  // C=1, T=1: 70% outcome
/// ]).unwrap()).unwrap();
///
/// // Adjust for confounder to get causal effect
/// let p_y_do_t = backdoor_adjustment(&net, 1, 1, 2, &[0]).unwrap();
/// ```
pub fn backdoor_adjustment(
    net: &BayesNet,
    x_var: usize,
    x_val: usize,
    y_var: usize,
    adjustment_set: &[usize],
) -> Result<Dist, ProbError> {
    let n_y_states = net.var_states[y_var];
    let mut result = vec![0.0; n_y_states];

    // Get states for adjustment variables
    let adj_states: Vec<usize> = adjustment_set.iter().map(|&v| net.var_states[v]).collect();
    let total_adj: usize = if adj_states.is_empty() {
        1
    } else {
        adj_states.iter().product()
    };

    for adj_idx in 0..total_adj {
        // Decode adjustment values
        let adj_values = decode_assignment(adj_idx, &adj_states);

        // Build evidence: X=x, Z=z
        let mut evidence = HashMap::new();
        evidence.insert(x_var, x_val);
        for (i, &z) in adjustment_set.iter().enumerate() {
            evidence.insert(z, adj_values[i]);
        }

        // P(Y | X=x, Z=z)
        let p_y_given_xz = match net.query(y_var, &evidence) {
            Ok(d) => d,
            Err(_) => continue, // Zero probability assignment
        };

        // P(Z=z) - marginal of adjustment set
        let p_z = compute_joint_marginal_prob(net, adjustment_set, &adj_values);

        // Accumulate: P(Y | X=x, Z=z) · P(Z=z)
        for (y, &p_y) in p_y_given_xz.p.iter().enumerate() {
            result[y] += p_y * p_z;
        }
    }

    Dist::from_weights(result)
}

/// Decode an index into assignment values.
fn decode_assignment(mut idx: usize, states: &[usize]) -> Vec<usize> {
    if states.is_empty() {
        return vec![];
    }
    let mut values = vec![0; states.len()];
    for (i, val) in values.iter_mut().enumerate() {
        *val = idx % states[i];
        idx /= states[i];
    }
    values
}

/// Compute P(Z₁=z₁, Z₂=z₂, ...) for a set of variables.
fn compute_joint_marginal_prob(net: &BayesNet, vars: &[usize], values: &[usize]) -> f32 {
    if vars.is_empty() {
        return 1.0;
    }

    let joint = net.full_joint();
    let mut prob = 0.0;

    for (idx, &p) in joint.probs.iter().enumerate() {
        let assignment = joint.decode(idx);
        let matches = vars
            .iter()
            .zip(values.iter())
            .all(|(&v, &val)| assignment[v] == val);
        if matches {
            prob += p;
        }
    }

    prob
}

/// Result of causal effect estimation.
#[derive(Debug, Clone)]
pub struct CausalEffect {
    /// P(Y | do(X=1)) - distribution under treatment
    pub p_y_do_1: Dist,
    /// P(Y | do(X=0)) - distribution under control
    pub p_y_do_0: Dist,
    /// Average Treatment Effect
    pub ate: f32,
    /// P(Y=1 | do(X=1)) - probability of positive outcome under treatment
    pub risk_treated: f32,
    /// P(Y=1 | do(X=0)) - probability of positive outcome under control
    pub risk_control: f32,
    /// Relative Risk = P(Y=1|do(X=1)) / P(Y=1|do(X=0))
    pub relative_risk: f32,
}

/// Compute comprehensive causal effect statistics.
///
/// For binary treatment and outcome, computes:
/// - ATE (Average Treatment Effect)
/// - Risk under treatment and control
/// - Relative Risk
pub fn causal_effect(
    net: &BayesNet,
    treatment_var: usize,
    outcome_var: usize,
) -> Result<CausalEffect, ProbError> {
    let p_y_do_1 = interventional_query(net, outcome_var, treatment_var, 1)?;
    let p_y_do_0 = interventional_query(net, outcome_var, treatment_var, 0)?;

    let risk_treated = if p_y_do_1.p.len() > 1 {
        p_y_do_1.p[1]
    } else {
        0.0
    };
    let risk_control = if p_y_do_0.p.len() > 1 {
        p_y_do_0.p[1]
    } else {
        0.0
    };

    let ate = risk_treated - risk_control;
    let relative_risk = if risk_control > 1e-10 {
        risk_treated / risk_control
    } else {
        f32::INFINITY
    };

    Ok(CausalEffect {
        p_y_do_1,
        p_y_do_0,
        ate,
        risk_treated,
        risk_control,
        relative_risk,
    })
}

/// Compare observational and interventional distributions.
///
/// Returns the difference between P(Y | X=x) and P(Y | do(X=x)),
/// which reveals confounding.
pub fn confounding_bias(
    net: &BayesNet,
    x_var: usize,
    x_val: usize,
    y_var: usize,
) -> Result<f32, ProbError> {
    // Observational: P(Y=1 | X=x)
    let mut evidence = HashMap::new();
    evidence.insert(x_var, x_val);
    let p_y_obs = net.query(y_var, &evidence)?;

    // Interventional: P(Y=1 | do(X=x))
    let p_y_int = interventional_query(net, y_var, x_var, x_val)?;

    // Bias = P(Y=1|X=x) - P(Y=1|do(X=x))
    let obs_prob = if p_y_obs.p.len() > 1 {
        p_y_obs.p[1]
    } else {
        0.0
    };
    let int_prob = if p_y_int.p.len() > 1 {
        p_y_int.p[1]
    } else {
        0.0
    };

    Ok(obs_prob - int_prob)
}

/// Create a smoking/cancer example network with confounding.
///
/// Structure:
/// ```text
///        Genotype (G)
///        ↙        ↘
///   Smoking (S) → Tar (T) → Cancer (C)
/// ```
///
/// - G: genetic predisposition (affects both smoking tendency and cancer risk)
/// - S: smoking behavior
/// - T: tar deposits in lungs
/// - C: lung cancer
pub fn smoking_cancer_network() -> BayesNet {
    use crate::kernel::Kernel;

    let net = BayesNet::new(vec![2, 2, 2, 2]); // G, S, T, C
    let net = net.with_names(vec!["Genotype", "Smoking", "Tar", "Cancer"]);
    let mut net = net;

    // P(G) - genetic predisposition
    net.add_prior(0, &Dist::new(vec![0.7, 0.3]).unwrap())
        .unwrap();

    // P(S|G) - smoking depends on genes
    net.add_conditional(
        1,
        vec![0],
        Kernel::new(vec![
            vec![0.8, 0.2], // G=0: 20% smoke
            vec![0.3, 0.7], // G=1: 70% smoke
        ])
        .unwrap(),
    )
    .unwrap();

    // P(T|S) - tar depends on smoking
    net.add_conditional(
        2,
        vec![1],
        Kernel::new(vec![
            vec![0.95, 0.05], // S=0: 5% tar
            vec![0.2, 0.8],   // S=1: 80% tar
        ])
        .unwrap(),
    )
    .unwrap();

    // P(C|G,T) - cancer depends on genes AND tar
    // Row-major: G*2 + T
    net.add_conditional(
        3,
        vec![0, 2],
        Kernel::new(vec![
            vec![0.98, 0.02], // G=0, T=0: 2% cancer
            vec![0.85, 0.15], // G=0, T=1: 15% cancer
            vec![0.90, 0.10], // G=1, T=0: 10% cancer
            vec![0.60, 0.40], // G=1, T=1: 40% cancer
        ])
        .unwrap(),
    )
    .unwrap();

    net
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sprinkler_network;
    use crate::PROB_TOLERANCE;

    #[test]
    fn test_intervention_basic() {
        let net = sprinkler_network();

        // Intervene: do(Sprinkler=On)
        let intervened = net.intervene(1, 1);

        // Sprinkler should now be deterministically On
        let p_sprinkler = intervened.marginal(1).unwrap();
        assert!((p_sprinkler.p[0] - 0.0).abs() < PROB_TOLERANCE);
        assert!((p_sprinkler.p[1] - 1.0).abs() < PROB_TOLERANCE);

        // Cloudy should be unaffected (no causal path from Sprinkler)
        let p_cloudy = intervened.marginal(0).unwrap();
        assert!((p_cloudy.p[0] - 0.5).abs() < PROB_TOLERANCE);
    }

    #[test]
    fn test_observational_vs_interventional() {
        let net = sprinkler_network();

        // Observational: P(Cloudy | Sprinkler=On)
        let mut evidence = HashMap::new();
        evidence.insert(1, 1);
        let p_cloudy_obs = net.query(0, &evidence).unwrap();

        // Interventional: P(Cloudy | do(Sprinkler=On))
        let intervened = net.intervene(1, 1);
        let p_cloudy_int = intervened.marginal(0).unwrap();

        // These should differ!
        // Observing Sprinkler=On tells us about Cloudy (common cause)
        // Intervening do(Sprinkler=On) doesn't affect Cloudy
        assert!((p_cloudy_obs.p[1] - p_cloudy_int.p[1]).abs() > 0.01);
    }

    #[test]
    fn test_smoking_confounding() {
        let net = smoking_cancer_network();

        // Observational P(Cancer | Smoking=1)
        let mut evidence = HashMap::new();
        evidence.insert(1, 1);
        let p_cancer_obs = net.query(3, &evidence).unwrap();

        // Interventional P(Cancer | do(Smoking=1))
        let p_cancer_int = interventional_query(&net, 3, 1, 1).unwrap();

        // Observational should be HIGHER due to confounding
        // (Smokers have the gene which independently causes cancer)
        assert!(
            p_cancer_obs.p[1] > p_cancer_int.p[1],
            "Confounding should inflate observational estimate: obs={}, int={}",
            p_cancer_obs.p[1],
            p_cancer_int.p[1]
        );
    }

    #[test]
    fn test_average_treatment_effect() {
        let net = smoking_cancer_network();

        let ate = average_treatment_effect(&net, 1, 3).unwrap();

        // Smoking should increase cancer risk (positive ATE)
        assert!(ate > 0.0, "ATE should be positive: {}", ate);
    }

    #[test]
    fn test_backdoor_adjustment() {
        let net = smoking_cancer_network();

        // Adjust for Genotype (the confounder)
        let p_y_adj = backdoor_adjustment(&net, 1, 1, 3, &[0]).unwrap();

        // This should match the interventional distribution
        let p_y_int = interventional_query(&net, 3, 1, 1).unwrap();

        assert!(
            (p_y_adj.p[1] - p_y_int.p[1]).abs() < 0.01,
            "Backdoor adjustment should match intervention: adj={}, int={}",
            p_y_adj.p[1],
            p_y_int.p[1]
        );
    }

    #[test]
    fn test_confounding_bias() {
        let net = smoking_cancer_network();

        let bias = confounding_bias(&net, 1, 1, 3).unwrap();

        // Bias should be positive (observational overestimates)
        assert!(bias > 0.0, "Confounding bias should be positive: {}", bias);
    }
}
