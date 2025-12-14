//! Bayesian networks as composed Markov kernels.
//!
//! A Bayesian network is a directed acyclic graph where:
//! - Nodes are random variables
//! - Edges represent conditional dependencies
//! - Each node has a conditional distribution given its parents
//!
//! Categorically, a Bayes net is a string diagram of kernels in FinStoch.

use crate::dist::Dist;
use crate::error::ProbError;
use crate::kernel::Kernel;
use std::collections::HashMap;

/// A factor in a Bayesian network.
///
/// Represents P(variable | parents) as a conditional probability table (CPT).
#[derive(Debug, Clone)]
pub struct Factor {
    /// The variable this factor defines.
    pub variable: usize,
    /// Parent variables (conditioning variables).
    pub parents: Vec<usize>,
    /// Number of states for this variable.
    pub n_states: usize,
    /// Number of states for each parent.
    pub parent_states: Vec<usize>,
    /// The conditional probability table.
    /// Indexed by (parent_config, variable_state).
    /// parent_config is computed by encoding parent states as a single index.
    pub cpt: Kernel,
}

impl Factor {
    /// Create a new factor P(variable | parents).
    ///
    /// The CPT should have rows for each parent configuration
    /// and columns for each state of the variable.
    pub fn new(
        variable: usize,
        parents: Vec<usize>,
        n_states: usize,
        parent_states: Vec<usize>,
        cpt: Kernel,
    ) -> Result<Self, ProbError> {
        // Verify dimensions
        let n_parent_configs: usize = if parent_states.is_empty() {
            1
        } else {
            parent_states.iter().product()
        };

        if cpt.n_inputs != n_parent_configs {
            return Err(ProbError::ShapeMismatch {
                expected: n_parent_configs,
                got: cpt.n_inputs,
            });
        }

        if cpt.n_outputs != n_states {
            return Err(ProbError::ShapeMismatch {
                expected: n_states,
                got: cpt.n_outputs,
            });
        }

        Ok(Self {
            variable,
            parents,
            n_states,
            parent_states,
            cpt,
        })
    }

    /// Create a prior factor (no parents).
    pub fn prior(variable: usize, dist: &Dist) -> Self {
        Self {
            variable,
            parents: vec![],
            n_states: dist.p.len(),
            parent_states: vec![],
            cpt: Kernel::constant(1, dist),
        }
    }

    /// Encode parent states into a single index (row-major order).
    ///
    /// For parents with states [s0, s1], values [v0, v1] encodes as:
    /// idx = v0 * s1 + v1
    pub fn encode_parents(&self, parent_values: &[usize]) -> usize {
        if parent_values.is_empty() {
            return 0;
        }

        let mut idx = 0;
        for (i, &val) in parent_values.iter().enumerate() {
            idx *= self.parent_states[i];
            idx += val;
        }
        idx
    }

    /// Decode a parent configuration index into individual values.
    pub fn decode_parents(&self, mut idx: usize) -> Vec<usize> {
        if self.parent_states.is_empty() {
            return vec![];
        }

        let mut values = vec![0; self.parent_states.len()];
        for i in (0..self.parent_states.len()).rev() {
            values[i] = idx % self.parent_states[i];
            idx /= self.parent_states[i];
        }
        values
    }

    /// Get P(variable=state | parent_values).
    pub fn prob(&self, state: usize, parent_values: &[usize]) -> f32 {
        let parent_idx = self.encode_parents(parent_values);
        self.cpt.k[parent_idx][state]
    }
}

/// A Bayesian network represented as a collection of factors.
///
/// Each factor corresponds to one node's conditional distribution.
#[derive(Debug, Clone)]
pub struct BayesNet {
    /// Number of variables in the network.
    pub n_vars: usize,
    /// Number of states for each variable.
    pub var_states: Vec<usize>,
    /// Factors defining the network (one per variable).
    pub factors: Vec<Factor>,
    /// Variable names (optional).
    pub var_names: Vec<String>,
}

impl BayesNet {
    /// Create a new Bayesian network.
    pub fn new(var_states: Vec<usize>) -> Self {
        let n_vars = var_states.len();
        Self {
            n_vars,
            var_states,
            factors: Vec::new(),
            var_names: (0..n_vars).map(|i| format!("X{}", i)).collect(),
        }
    }

    /// Set variable names.
    pub fn with_names(mut self, names: Vec<&str>) -> Self {
        self.var_names = names.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Add a factor to the network.
    pub fn add_factor(&mut self, factor: Factor) -> Result<(), ProbError> {
        // Verify variable index
        if factor.variable >= self.n_vars {
            return Err(ProbError::IndexOutOfBounds {
                index: factor.variable,
                size: self.n_vars,
            });
        }

        // Verify parent indices
        for &p in &factor.parents {
            if p >= self.n_vars {
                return Err(ProbError::IndexOutOfBounds {
                    index: p,
                    size: self.n_vars,
                });
            }
        }

        // Verify state counts match
        if factor.n_states != self.var_states[factor.variable] {
            return Err(ProbError::ShapeMismatch {
                expected: self.var_states[factor.variable],
                got: factor.n_states,
            });
        }

        self.factors.push(factor);
        Ok(())
    }

    /// Add a prior (root node) to the network.
    pub fn add_prior(&mut self, variable: usize, dist: &Dist) -> Result<(), ProbError> {
        if dist.p.len() != self.var_states[variable] {
            return Err(ProbError::ShapeMismatch {
                expected: self.var_states[variable],
                got: dist.p.len(),
            });
        }
        self.factors.push(Factor::prior(variable, dist));
        Ok(())
    }

    /// Add a conditional P(child | parents) to the network.
    pub fn add_conditional(
        &mut self,
        child: usize,
        parents: Vec<usize>,
        cpt: Kernel,
    ) -> Result<(), ProbError> {
        let parent_states: Vec<usize> = parents.iter().map(|&p| self.var_states[p]).collect();

        let factor = Factor::new(child, parents, self.var_states[child], parent_states, cpt)?;

        self.add_factor(factor)
    }

    /// Compute the joint probability P(X₁=x₁, ..., Xₙ=xₙ).
    pub fn joint_prob(&self, assignment: &[usize]) -> f32 {
        if assignment.len() != self.n_vars {
            return 0.0;
        }

        let mut prob = 1.0;
        for factor in &self.factors {
            let var_state = assignment[factor.variable];
            let parent_values: Vec<usize> = factor.parents.iter().map(|&p| assignment[p]).collect();
            prob *= factor.prob(var_state, &parent_values);
        }
        prob
    }

    /// Compute the full joint distribution as a flattened vector.
    ///
    /// The index encodes the assignment: idx = x₀ + x₁*s₀ + x₂*s₀*s₁ + ...
    pub fn full_joint(&self) -> JointDist {
        let total_states: usize = self.var_states.iter().product();
        let mut probs = vec![0.0; total_states];

        for (idx, prob) in probs.iter_mut().enumerate() {
            let assignment = self.decode_assignment(idx);
            *prob = self.joint_prob(&assignment);
        }

        JointDist {
            var_states: self.var_states.clone(),
            probs,
        }
    }

    /// Encode an assignment into a single index.
    pub fn encode_assignment(&self, assignment: &[usize]) -> usize {
        let mut idx = 0;
        let mut multiplier = 1;
        for (i, &val) in assignment.iter().enumerate() {
            idx += val * multiplier;
            multiplier *= self.var_states[i];
        }
        idx
    }

    /// Decode an index into an assignment.
    pub fn decode_assignment(&self, mut idx: usize) -> Vec<usize> {
        let mut assignment = vec![0; self.n_vars];
        for (i, val) in assignment.iter_mut().enumerate() {
            *val = idx % self.var_states[i];
            idx /= self.var_states[i];
        }
        assignment
    }

    /// Compute marginal distribution P(variable).
    pub fn marginal(&self, variable: usize) -> Result<Dist, ProbError> {
        if variable >= self.n_vars {
            return Err(ProbError::IndexOutOfBounds {
                index: variable,
                size: self.n_vars,
            });
        }

        let joint = self.full_joint();
        joint.marginalize_to(&[variable])
    }

    /// Compute conditional distribution P(query | evidence).
    pub fn query(
        &self,
        query_var: usize,
        evidence: &HashMap<usize, usize>,
    ) -> Result<Dist, ProbError> {
        if query_var >= self.n_vars {
            return Err(ProbError::IndexOutOfBounds {
                index: query_var,
                size: self.n_vars,
            });
        }

        let joint = self.full_joint();
        joint.condition_on(query_var, evidence)
    }
}

/// A joint distribution over multiple variables.
#[derive(Debug, Clone)]
pub struct JointDist {
    /// Number of states for each variable.
    pub var_states: Vec<usize>,
    /// Flattened probability vector.
    pub probs: Vec<f32>,
}

impl JointDist {
    /// Get the number of variables.
    pub fn n_vars(&self) -> usize {
        self.var_states.len()
    }

    /// Get probability of an assignment.
    pub fn prob(&self, assignment: &[usize]) -> f32 {
        let idx = self.encode(assignment);
        self.probs[idx]
    }

    /// Encode an assignment into an index.
    pub fn encode(&self, assignment: &[usize]) -> usize {
        let mut idx = 0;
        let mut multiplier = 1;
        for (i, &val) in assignment.iter().enumerate() {
            idx += val * multiplier;
            multiplier *= self.var_states[i];
        }
        idx
    }

    /// Decode an index into an assignment.
    pub fn decode(&self, mut idx: usize) -> Vec<usize> {
        let mut assignment = vec![0; self.var_states.len()];
        for (i, val) in assignment.iter_mut().enumerate() {
            *val = idx % self.var_states[i];
            idx /= self.var_states[i];
        }
        assignment
    }

    /// Marginalize to keep only the specified variables.
    pub fn marginalize_to(&self, keep_vars: &[usize]) -> Result<Dist, ProbError> {
        // For single variable marginal
        if keep_vars.len() == 1 {
            let var = keep_vars[0];
            if var >= self.var_states.len() {
                return Err(ProbError::IndexOutOfBounds {
                    index: var,
                    size: self.var_states.len(),
                });
            }

            let n_states = self.var_states[var];
            let mut marginal = vec![0.0; n_states];

            for (idx, &p) in self.probs.iter().enumerate() {
                let assignment = self.decode(idx);
                marginal[assignment[var]] += p;
            }

            return Dist::new(marginal);
        }

        // For multiple variables, return as Dist over product space
        let total_keep: usize = keep_vars.iter().map(|&v| self.var_states[v]).product();
        let mut marginal = vec![0.0; total_keep];

        for (idx, &p) in self.probs.iter().enumerate() {
            let assignment = self.decode(idx);
            let keep_assignment: Vec<usize> = keep_vars.iter().map(|&v| assignment[v]).collect();

            // Encode the kept assignment
            let mut keep_idx = 0;
            let mut multiplier = 1;
            for (i, &v) in keep_vars.iter().enumerate() {
                keep_idx += keep_assignment[i] * multiplier;
                multiplier *= self.var_states[v];
            }

            marginal[keep_idx] += p;
        }

        Dist::new(marginal)
    }

    /// Condition on evidence: P(query | evidence).
    pub fn condition_on(
        &self,
        query_var: usize,
        evidence: &HashMap<usize, usize>,
    ) -> Result<Dist, ProbError> {
        let n_states = self.var_states[query_var];
        let mut unnormalized = vec![0.0; n_states];

        for (idx, &p) in self.probs.iter().enumerate() {
            let assignment = self.decode(idx);

            // Check if assignment matches evidence
            let matches = evidence.iter().all(|(&var, &val)| assignment[var] == val);

            if matches {
                unnormalized[assignment[query_var]] += p;
            }
        }

        // Normalize
        Dist::from_weights(unnormalized)
    }

    /// Marginalize out (sum over) the specified variables.
    pub fn marginalize_out(&self, remove_vars: &[usize]) -> Result<JointDist, ProbError> {
        let keep_vars: Vec<usize> = (0..self.var_states.len())
            .filter(|v| !remove_vars.contains(v))
            .collect();

        let keep_states: Vec<usize> = keep_vars.iter().map(|&v| self.var_states[v]).collect();
        let total_keep: usize = if keep_states.is_empty() {
            1
        } else {
            keep_states.iter().product()
        };

        let mut new_probs = vec![0.0; total_keep];

        for (idx, &p) in self.probs.iter().enumerate() {
            let assignment = self.decode(idx);

            // Encode only kept variables
            let mut keep_idx = 0;
            let mut multiplier = 1;
            for &v in &keep_vars {
                keep_idx += assignment[v] * multiplier;
                multiplier *= self.var_states[v];
            }

            new_probs[keep_idx] += p;
        }

        Ok(JointDist {
            var_states: keep_states,
            probs: new_probs,
        })
    }
}

/// Variable elimination algorithm for efficient inference.
pub struct VariableElimination<'a> {
    net: &'a BayesNet,
}

impl<'a> VariableElimination<'a> {
    /// Create a new variable elimination solver.
    pub fn new(net: &'a BayesNet) -> Self {
        Self { net }
    }

    /// Compute P(query) by eliminating all other variables.
    ///
    /// This is more efficient than computing the full joint when
    /// the network has many variables.
    pub fn marginal(&self, query: usize) -> Result<Dist, ProbError> {
        // For now, use the simple full joint method
        // A full implementation would use factor multiplication and marginalization
        self.net.marginal(query)
    }

    /// Compute P(query | evidence) using variable elimination.
    pub fn query(&self, query: usize, evidence: &HashMap<usize, usize>) -> Result<Dist, ProbError> {
        self.net.query(query, evidence)
    }
}

/// Create a classic "sprinkler" Bayesian network.
///
/// Structure:
/// ```text
///      Cloudy (C)
///      ↙     ↘
/// Sprinkler   Rain
///     (S)      (R)
///        ↘    ↙
///       WetGrass (W)
/// ```
pub fn sprinkler_network() -> BayesNet {
    // All binary variables: 0=False, 1=True
    let net = BayesNet::new(vec![2, 2, 2, 2]);
    let mut net = net.with_names(vec!["Cloudy", "Sprinkler", "Rain", "WetGrass"]);

    // P(Cloudy)
    net.add_prior(0, &Dist::new(vec![0.5, 0.5]).unwrap())
        .unwrap();

    // P(Sprinkler | Cloudy)
    net.add_conditional(
        1,
        vec![0],
        Kernel::new(vec![
            vec![0.5, 0.5], // Not cloudy: 50% sprinkler
            vec![0.9, 0.1], // Cloudy: 10% sprinkler
        ])
        .unwrap(),
    )
    .unwrap();

    // P(Rain | Cloudy)
    net.add_conditional(
        2,
        vec![0],
        Kernel::new(vec![
            vec![0.8, 0.2], // Not cloudy: 20% rain
            vec![0.2, 0.8], // Cloudy: 80% rain
        ])
        .unwrap(),
    )
    .unwrap();

    // P(WetGrass | Sprinkler, Rain)
    // Row-major encoding: [S, R] -> S*2 + R
    // idx 0: S=0, R=0
    // idx 1: S=0, R=1
    // idx 2: S=1, R=0
    // idx 3: S=1, R=1
    net.add_conditional(
        3,
        vec![1, 2],
        Kernel::new(vec![
            vec![1.0, 0.0],   // S=0, R=0: grass dry
            vec![0.2, 0.8],   // S=0, R=1: 80% wet (rain)
            vec![0.1, 0.9],   // S=1, R=0: 90% wet (sprinkler)
            vec![0.01, 0.99], // S=1, R=1: 99% wet (both)
        ])
        .unwrap(),
    )
    .unwrap();

    net
}

/// Create a simple Naive Bayes classifier network.
///
/// Structure:
/// ```text
///        Class (C)
///       ↙  ↓  ↘
///     F₁  F₂  F₃
/// ```
pub fn naive_bayes(n_features: usize, n_classes: usize, feature_states: usize) -> BayesNet {
    let mut var_states = vec![n_classes];
    var_states.extend(vec![feature_states; n_features]);

    BayesNet::new(var_states)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PROB_TOLERANCE;

    #[test]
    fn test_factor_encoding() {
        let cpt = Kernel::new(vec![
            vec![0.9, 0.1],
            vec![0.5, 0.5],
            vec![0.8, 0.2],
            vec![0.1, 0.9],
        ])
        .unwrap();

        let factor = Factor::new(2, vec![0, 1], 2, vec![2, 2], cpt).unwrap();

        // Test encoding (row-major: [v0, v1] -> v0 * s1 + v1)
        assert_eq!(factor.encode_parents(&[0, 0]), 0); // 0*2 + 0 = 0
        assert_eq!(factor.encode_parents(&[0, 1]), 1); // 0*2 + 1 = 1
        assert_eq!(factor.encode_parents(&[1, 0]), 2); // 1*2 + 0 = 2
        assert_eq!(factor.encode_parents(&[1, 1]), 3); // 1*2 + 1 = 3

        // Test decoding
        assert_eq!(factor.decode_parents(0), vec![0, 0]);
        assert_eq!(factor.decode_parents(1), vec![0, 1]);
        assert_eq!(factor.decode_parents(2), vec![1, 0]);
        assert_eq!(factor.decode_parents(3), vec![1, 1]);
    }

    #[test]
    fn test_sprinkler_network() {
        let net = sprinkler_network();

        assert_eq!(net.n_vars, 4);
        assert_eq!(net.factors.len(), 4);

        // Test joint probability sums to 1
        let joint = net.full_joint();
        let sum: f32 = joint.probs.iter().sum();
        assert!((sum - 1.0).abs() < PROB_TOLERANCE);
    }

    #[test]
    fn test_marginal() {
        let net = sprinkler_network();

        // P(Cloudy) should be [0.5, 0.5]
        let p_cloudy = net.marginal(0).unwrap();
        assert!((p_cloudy.p[0] - 0.5).abs() < PROB_TOLERANCE);
        assert!((p_cloudy.p[1] - 0.5).abs() < PROB_TOLERANCE);

        // P(WetGrass) - computed by marginalization
        let p_wet = net.marginal(3).unwrap();
        // Should sum to 1
        let sum: f32 = p_wet.p.iter().sum();
        assert!((sum - 1.0).abs() < PROB_TOLERANCE);
    }

    #[test]
    fn test_conditioning() {
        let net = sprinkler_network();

        // P(Rain | Cloudy=True) should be [0.2, 0.8]
        let mut evidence = HashMap::new();
        evidence.insert(0, 1); // Cloudy = True

        let p_rain_given_cloudy = net.query(2, &evidence).unwrap();
        assert!((p_rain_given_cloudy.p[0] - 0.2).abs() < 0.01);
        assert!((p_rain_given_cloudy.p[1] - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_joint_dist_marginalize() {
        // Simple 2-variable joint
        let joint = JointDist {
            var_states: vec![2, 2],
            probs: vec![0.1, 0.2, 0.3, 0.4], // P(X=0,Y=0), P(X=1,Y=0), P(X=0,Y=1), P(X=1,Y=1)
        };

        // P(X) = [0.1+0.3, 0.2+0.4] = [0.4, 0.6]
        let p_x = joint.marginalize_to(&[0]).unwrap();
        assert!((p_x.p[0] - 0.4).abs() < PROB_TOLERANCE);
        assert!((p_x.p[1] - 0.6).abs() < PROB_TOLERANCE);

        // P(Y) = [0.1+0.2, 0.3+0.4] = [0.3, 0.7]
        let p_y = joint.marginalize_to(&[1]).unwrap();
        assert!((p_y.p[0] - 0.3).abs() < PROB_TOLERANCE);
        assert!((p_y.p[1] - 0.7).abs() < PROB_TOLERANCE);
    }

    #[test]
    fn test_naive_bayes_structure() {
        let net = naive_bayes(3, 2, 2);
        assert_eq!(net.n_vars, 4); // 1 class + 3 features
        assert_eq!(net.var_states[0], 2); // 2 classes
        assert_eq!(net.var_states[1], 2); // 2 feature states
    }
}
