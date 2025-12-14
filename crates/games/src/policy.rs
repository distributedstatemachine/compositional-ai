//! Policies: Observation → Action mappings.
//!
//! A policy decides what action to take given an observation.
//! This is one of the swappable components in open games.
//!
//! # Compositional Design
//!
//! Policies are morphisms: `Obs → Act`
//!
//! ```text
//!        ┌──────────┐
//!  Obs ─▶│  Policy  │─▶ Act
//!        └──────────┘
//! ```
//!
//! You can swap policies without changing the environment!

use std::marker::PhantomData;

/// A policy maps observations to actions.
///
/// This is a key swappable component: same environment, different policy
/// = different behavior.
pub trait Policy<Obs, Act> {
    /// Select an action given an observation.
    fn act(&self, obs: &Obs) -> Act;

    /// Optional: get action probabilities (for stochastic policies).
    fn action_probs(&self, _obs: &Obs) -> Option<Vec<f64>> {
        None
    }
}

// ============================================================================
// Random Policy
// ============================================================================

/// A policy that selects actions uniformly at random.
///
/// Useful as a baseline or for exploration.
#[derive(Debug, Clone)]
pub struct RandomPolicy {
    /// Number of possible actions
    num_actions: usize,
    /// Random state (simple LCG for reproducibility)
    seed: std::cell::Cell<u64>,
}

impl RandomPolicy {
    /// Create a new random policy with given number of actions.
    pub fn new(num_actions: usize) -> Self {
        Self {
            num_actions,
            seed: std::cell::Cell::new(42),
        }
    }

    /// Create with specific seed.
    pub fn with_seed(num_actions: usize, seed: u64) -> Self {
        Self {
            num_actions,
            seed: std::cell::Cell::new(seed),
        }
    }

    /// Simple random number generator (LCG).
    fn next_random(&self) -> f64 {
        let s = self.seed.get();
        let new_seed = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.seed.set(new_seed);
        (new_seed >> 33) as f64 / (1u64 << 31) as f64
    }
}

impl<Obs> Policy<Obs, usize> for RandomPolicy {
    fn act(&self, _obs: &Obs) -> usize {
        let r = self.next_random();
        (r * self.num_actions as f64) as usize
    }

    fn action_probs(&self, _obs: &Obs) -> Option<Vec<f64>> {
        let prob = 1.0 / self.num_actions as f64;
        Some(vec![prob; self.num_actions])
    }
}

// ============================================================================
// Greedy Policy
// ============================================================================

/// A policy that always selects the action with highest Q-value.
///
/// Uses a Q-function to evaluate state-action pairs.
#[derive(Debug, Clone)]
pub struct GreedyPolicy<Q> {
    /// Q-function: (state, action) → value
    q_function: Q,
    /// Number of actions
    num_actions: usize,
}

impl<Q> GreedyPolicy<Q> {
    /// Create a new greedy policy with given Q-function.
    pub fn new(q_function: Q, num_actions: usize) -> Self {
        Self {
            q_function,
            num_actions,
        }
    }
}

impl<Obs, Q> Policy<Obs, usize> for GreedyPolicy<Q>
where
    Q: Fn(&Obs, usize) -> f64,
{
    fn act(&self, obs: &Obs) -> usize {
        let mut best_action = 0;
        let mut best_value = f64::NEG_INFINITY;

        for action in 0..self.num_actions {
            let value = (self.q_function)(obs, action);
            if value > best_value {
                best_value = value;
                best_action = action;
            }
        }

        best_action
    }
}

// ============================================================================
// Epsilon-Greedy Policy
// ============================================================================

/// A policy that selects greedy action with probability (1-ε),
/// and random action with probability ε.
#[derive(Debug, Clone)]
pub struct EpsilonGreedyPolicy<Q> {
    /// Q-function
    q_function: Q,
    /// Number of actions
    num_actions: usize,
    /// Exploration probability
    epsilon: f64,
    /// Random seed
    seed: std::cell::Cell<u64>,
}

impl<Q> EpsilonGreedyPolicy<Q> {
    /// Create a new epsilon-greedy policy.
    pub fn new(q_function: Q, num_actions: usize, epsilon: f64) -> Self {
        Self {
            q_function,
            num_actions,
            epsilon: epsilon.clamp(0.0, 1.0),
            seed: std::cell::Cell::new(42),
        }
    }

    /// Simple random number generator.
    fn next_random(&self) -> f64 {
        let s = self.seed.get();
        let new_seed = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.seed.set(new_seed);
        (new_seed >> 33) as f64 / (1u64 << 31) as f64
    }
}

impl<Obs, Q> Policy<Obs, usize> for EpsilonGreedyPolicy<Q>
where
    Q: Fn(&Obs, usize) -> f64,
{
    fn act(&self, obs: &Obs) -> usize {
        if self.next_random() < self.epsilon {
            // Random action
            (self.next_random() * self.num_actions as f64) as usize
        } else {
            // Greedy action
            let mut best_action = 0;
            let mut best_value = f64::NEG_INFINITY;

            for action in 0..self.num_actions {
                let value = (self.q_function)(obs, action);
                if value > best_value {
                    best_value = value;
                    best_action = action;
                }
            }

            best_action
        }
    }

    fn action_probs(&self, obs: &Obs) -> Option<Vec<f64>> {
        let mut probs = vec![self.epsilon / self.num_actions as f64; self.num_actions];

        // Find greedy action
        let mut best_action = 0;
        let mut best_value = f64::NEG_INFINITY;

        for action in 0..self.num_actions {
            let value = (self.q_function)(obs, action);
            if value > best_value {
                best_value = value;
                best_action = action;
            }
        }

        probs[best_action] += 1.0 - self.epsilon;
        Some(probs)
    }
}

// ============================================================================
// Constant Policy
// ============================================================================

/// A policy that always returns the same action.
///
/// Useful for testing and as a baseline.
#[derive(Debug, Clone)]
pub struct ConstantPolicy<Act> {
    action: Act,
}

impl<Act: Clone> ConstantPolicy<Act> {
    /// Create a policy that always returns the given action.
    pub fn new(action: Act) -> Self {
        Self { action }
    }
}

impl<Obs, Act: Clone> Policy<Obs, Act> for ConstantPolicy<Act> {
    fn act(&self, _obs: &Obs) -> Act {
        self.action.clone()
    }
}

// ============================================================================
// Function Policy
// ============================================================================

/// A policy defined by a function.
///
/// Wraps any `Fn(&Obs) -> Act` as a policy.
pub struct FnPolicy<F, Obs, Act> {
    f: F,
    _phantom: PhantomData<(Obs, Act)>,
}

impl<F, Obs, Act> FnPolicy<F, Obs, Act> {
    /// Create a policy from a function.
    pub fn new(f: F) -> Self {
        Self {
            f,
            _phantom: PhantomData,
        }
    }
}

impl<F, Obs, Act> Policy<Obs, Act> for FnPolicy<F, Obs, Act>
where
    F: Fn(&Obs) -> Act,
{
    fn act(&self, obs: &Obs) -> Act {
        (self.f)(obs)
    }
}

// ============================================================================
// Composed Policy
// ============================================================================

/// Compose two policies: first transforms observation, second decides action.
///
/// This enables hierarchical policies and observation preprocessing.
pub struct ComposedPolicy<P1, P2, Obs, Mid, Act> {
    /// First policy: Obs → Mid
    first: P1,
    /// Second policy: Mid → Act
    second: P2,
    _phantom: PhantomData<(Obs, Mid, Act)>,
}

impl<P1, P2, Obs, Mid, Act> ComposedPolicy<P1, P2, Obs, Mid, Act> {
    /// Create a composed policy.
    pub fn new(first: P1, second: P2) -> Self {
        Self {
            first,
            second,
            _phantom: PhantomData,
        }
    }
}

impl<P1, P2, Obs, Mid, Act> Policy<Obs, Act> for ComposedPolicy<P1, P2, Obs, Mid, Act>
where
    P1: Policy<Obs, Mid>,
    P2: Policy<Mid, Act>,
{
    fn act(&self, obs: &Obs) -> Act {
        let mid = self.first.act(obs);
        self.second.act(&mid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_policy() {
        let policy = RandomPolicy::new(4);
        let obs = 0;

        // Should produce valid actions
        for _ in 0..100 {
            let action = policy.act(&obs);
            assert!(action < 4);
        }
    }

    #[test]
    fn test_constant_policy() {
        let policy = ConstantPolicy::new(42usize);

        assert_eq!(policy.act(&"anything"), 42);
        assert_eq!(policy.act(&123), 42);
    }

    #[test]
    fn test_greedy_policy() {
        // Q-function that prefers action 2
        let q = |_obs: &i32, act: usize| -> f64 {
            if act == 2 {
                10.0
            } else {
                0.0
            }
        };

        let policy = GreedyPolicy::new(q, 4);
        assert_eq!(policy.act(&0), 2);
    }

    #[test]
    fn test_fn_policy() {
        let policy = FnPolicy::new(|obs: &i32| (*obs % 4) as usize);

        assert_eq!(policy.act(&0), 0);
        assert_eq!(policy.act(&5), 1);
        assert_eq!(policy.act(&10), 2);
    }

    #[test]
    fn test_epsilon_greedy_action_probs() {
        let q = |_obs: &i32, act: usize| -> f64 {
            if act == 0 {
                10.0
            } else {
                0.0
            }
        };

        let policy = EpsilonGreedyPolicy::new(q, 4, 0.2);
        let probs = policy.action_probs(&0).unwrap();

        // Action 0 should have probability 0.8 + 0.2/4 = 0.85
        assert!((probs[0] - 0.85).abs() < 1e-10);
        // Others should have 0.2/4 = 0.05
        assert!((probs[1] - 0.05).abs() < 1e-10);
    }
}
