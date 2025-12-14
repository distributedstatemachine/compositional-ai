//! Rollout: Execute a policy in an environment.
//!
//! The rollout function is the core composition operation: it combines
//! a policy and an environment to produce a trajectory.
//!
//! # Compositional Design
//!
//! ```text
//!    ┌──────────┐     ┌──────────┐
//!    │  Policy  │────▶│   Env    │
//!    └──────────┘     └──────────┘
//!         │                │
//!         │                │
//!         ▼                ▼
//!    ┌─────────────────────────────┐
//!    │        Trajectory           │
//!    │  [(s₀,a₀,r₀), (s₁,a₁,r₁),...]│
//!    └─────────────────────────────┘
//! ```
//!
//! The rollout is where policy meets environment!

use crate::env::{Env, Transition};
use crate::policy::Policy;

/// A single step in a trajectory.
#[derive(Debug, Clone)]
pub struct Step<State, Obs, Act> {
    /// State at this step
    pub state: State,
    /// Observation at this step
    pub observation: Obs,
    /// Action taken
    pub action: Act,
    /// Reward received
    pub reward: f64,
    /// Next state (after action)
    pub next_state: State,
    /// Next observation
    pub next_observation: Obs,
    /// Whether this step ended the episode
    pub done: bool,
}

/// A trajectory is a sequence of steps from a rollout.
#[derive(Debug, Clone)]
pub struct Trajectory<State, Obs, Act> {
    /// The steps in the trajectory
    pub steps: Vec<Step<State, Obs, Act>>,
}

impl<State, Obs, Act> Trajectory<State, Obs, Act> {
    /// Create a new empty trajectory.
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Add a step to the trajectory.
    pub fn push(&mut self, step: Step<State, Obs, Act>) {
        self.steps.push(step);
    }

    /// Get the total reward.
    pub fn total_reward(&self) -> f64 {
        self.steps.iter().map(|s| s.reward).sum()
    }

    /// Get the discounted reward with given discount factor.
    pub fn discounted_reward(&self, gamma: f64) -> f64 {
        let mut total = 0.0;
        let mut discount = 1.0;
        for step in &self.steps {
            total += discount * step.reward;
            discount *= gamma;
        }
        total
    }

    /// Get the length of the trajectory.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Check if trajectory is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Check if trajectory ended due to terminal state.
    pub fn is_terminal(&self) -> bool {
        self.steps.last().map(|s| s.done).unwrap_or(false)
    }

    /// Get rewards as a vector.
    pub fn rewards(&self) -> Vec<f64> {
        self.steps.iter().map(|s| s.reward).collect()
    }

    /// Get actions as a vector (if Act is Clone).
    pub fn actions(&self) -> Vec<Act>
    where
        Act: Clone,
    {
        self.steps.iter().map(|s| s.action.clone()).collect()
    }
}

impl<State, Obs, Act> Default for Trajectory<State, Obs, Act> {
    fn default() -> Self {
        Self::new()
    }
}

/// Execute a policy in an environment for at most T steps.
///
/// This is the core operation that composes policy and environment.
///
/// # Arguments
/// * `policy` - The policy to use for action selection
/// * `env` - The environment to interact with
/// * `max_steps` - Maximum number of steps to take
///
/// # Returns
/// A trajectory containing all steps taken
///
/// # Example
///
/// ```rust
/// use compositional_games::{rollout, Policy, Env};
/// use compositional_games::env::GridWorld;
/// use compositional_games::policy::RandomPolicy;
///
/// let env = GridWorld::new(5, 5);
/// let policy = RandomPolicy::new(4);
///
/// let trajectory = rollout(&policy, &env, 100);
/// println!("Steps: {}, Total reward: {:.2}",
///          trajectory.len(),
///          trajectory.total_reward());
/// ```
pub fn rollout<P, E>(policy: &P, env: &E, max_steps: usize) -> Trajectory<E::State, E::Obs, E::Act>
where
    E: Env,
    E::State: Clone,
    E::Obs: Clone,
    E::Act: Clone,
    P: Policy<E::Obs, E::Act>,
{
    let mut trajectory = Trajectory::new();

    // Reset environment
    let (mut state, mut obs) = env.reset();

    for _ in 0..max_steps {
        // Check if terminal
        if env.is_terminal(&state) {
            break;
        }

        // Get action from policy
        let action = policy.act(&obs);

        // Take step in environment
        match env.step(&state, &action) {
            Ok(Transition {
                next_state,
                observation: next_obs,
                reward,
                done,
            }) => {
                // Record step
                trajectory.push(Step {
                    state: state.clone(),
                    observation: obs.clone(),
                    action,
                    reward,
                    next_state: next_state.clone(),
                    next_observation: next_obs.clone(),
                    done,
                });

                // Update state
                state = next_state;
                obs = next_obs;

                if done {
                    break;
                }
            }
            Err(_) => break,
        }
    }

    trajectory
}

/// Execute multiple rollouts and return all trajectories.
///
/// Useful for collecting data or evaluating a policy.
pub fn rollout_batch<P, E>(
    policy: &P,
    env: &E,
    num_episodes: usize,
    max_steps: usize,
) -> Vec<Trajectory<E::State, E::Obs, E::Act>>
where
    E: Env,
    E::State: Clone,
    E::Obs: Clone,
    E::Act: Clone,
    P: Policy<E::Obs, E::Act>,
{
    (0..num_episodes)
        .map(|_| rollout(policy, env, max_steps))
        .collect()
}

/// Compute statistics over multiple rollouts.
#[derive(Debug, Clone)]
pub struct RolloutStats {
    /// Number of episodes
    pub num_episodes: usize,
    /// Mean total reward
    pub mean_reward: f64,
    /// Standard deviation of total reward
    pub std_reward: f64,
    /// Mean episode length
    pub mean_length: f64,
    /// Number of episodes that reached terminal state
    pub num_terminal: usize,
}

impl std::fmt::Display for RolloutStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Rollout Statistics ({} episodes):", self.num_episodes)?;
        writeln!(
            f,
            "  Mean reward: {:.4} ± {:.4}",
            self.mean_reward, self.std_reward
        )?;
        writeln!(f, "  Mean length: {:.2}", self.mean_length)?;
        writeln!(
            f,
            "  Terminal: {}/{} ({:.1}%)",
            self.num_terminal,
            self.num_episodes,
            100.0 * self.num_terminal as f64 / self.num_episodes as f64
        )
    }
}

/// Evaluate a policy by running multiple rollouts and computing statistics.
pub fn evaluate<P, E>(policy: &P, env: &E, num_episodes: usize, max_steps: usize) -> RolloutStats
where
    E: Env,
    E::State: Clone,
    E::Obs: Clone,
    E::Act: Clone,
    P: Policy<E::Obs, E::Act>,
{
    let trajectories = rollout_batch(policy, env, num_episodes, max_steps);

    let rewards: Vec<f64> = trajectories.iter().map(|t| t.total_reward()).collect();
    let lengths: Vec<f64> = trajectories.iter().map(|t| t.len() as f64).collect();
    let num_terminal = trajectories.iter().filter(|t| t.is_terminal()).count();

    let mean_reward = rewards.iter().sum::<f64>() / rewards.len() as f64;
    let mean_length = lengths.iter().sum::<f64>() / lengths.len() as f64;

    let variance = rewards
        .iter()
        .map(|r| (r - mean_reward).powi(2))
        .sum::<f64>()
        / rewards.len() as f64;
    let std_reward = variance.sqrt();

    RolloutStats {
        num_episodes,
        mean_reward,
        std_reward,
        mean_length,
        num_terminal,
    }
}

/// Compare two policies by running rollouts on the same environment.
#[derive(Debug, Clone)]
pub struct PolicyComparison {
    /// Stats for policy 1
    pub policy1_stats: RolloutStats,
    /// Stats for policy 2
    pub policy2_stats: RolloutStats,
}

impl PolicyComparison {
    /// Compute the difference in mean reward (policy1 - policy2).
    pub fn reward_difference(&self) -> f64 {
        self.policy1_stats.mean_reward - self.policy2_stats.mean_reward
    }

    /// Check if policy 1 is better (higher mean reward).
    pub fn policy1_better(&self) -> bool {
        self.reward_difference() > 0.0
    }
}

impl std::fmt::Display for PolicyComparison {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Policy Comparison:")?;
        writeln!(f, "\nPolicy 1:")?;
        writeln!(
            f,
            "  Mean reward: {:.4} ± {:.4}",
            self.policy1_stats.mean_reward, self.policy1_stats.std_reward
        )?;
        writeln!(f, "\nPolicy 2:")?;
        writeln!(
            f,
            "  Mean reward: {:.4} ± {:.4}",
            self.policy2_stats.mean_reward, self.policy2_stats.std_reward
        )?;
        writeln!(f, "\nDifference: {:.4}", self.reward_difference())?;
        writeln!(
            f,
            "Winner: Policy {}",
            if self.policy1_better() { 1 } else { 2 }
        )
    }
}

/// Compare two policies on the same environment.
pub fn compare_policies<P1, P2, E>(
    policy1: &P1,
    policy2: &P2,
    env: &E,
    num_episodes: usize,
    max_steps: usize,
) -> PolicyComparison
where
    E: Env,
    E::State: Clone,
    E::Obs: Clone,
    E::Act: Clone,
    P1: Policy<E::Obs, E::Act>,
    P2: Policy<E::Obs, E::Act>,
{
    PolicyComparison {
        policy1_stats: evaluate(policy1, env, num_episodes, max_steps),
        policy2_stats: evaluate(policy2, env, num_episodes, max_steps),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::GridWorld;
    use crate::policy::{ConstantPolicy, RandomPolicy};

    #[test]
    fn test_rollout_basic() {
        let env = GridWorld::new(3, 3);
        let policy = RandomPolicy::new(4);

        let trajectory = rollout(&policy, &env, 100);

        // Should have some steps
        assert!(!trajectory.is_empty());

        // Total reward should be computable
        let _total = trajectory.total_reward();
    }

    #[test]
    fn test_rollout_terminates() {
        // With constant "right" policy, should reach goal
        let env = GridWorld::new(3, 3).with_goal(2, 0);
        let policy = ConstantPolicy::new(3usize); // Always go right

        let trajectory = rollout(&policy, &env, 100);

        assert!(trajectory.is_terminal());
        assert!(trajectory.total_reward() > 0.0);
    }

    #[test]
    fn test_trajectory_discounted_reward() {
        let mut traj: Trajectory<(), (), ()> = Trajectory::new();
        traj.push(Step {
            state: (),
            observation: (),
            action: (),
            reward: 1.0,
            next_state: (),
            next_observation: (),
            done: false,
        });
        traj.push(Step {
            state: (),
            observation: (),
            action: (),
            reward: 1.0,
            next_state: (),
            next_observation: (),
            done: false,
        });
        traj.push(Step {
            state: (),
            observation: (),
            action: (),
            reward: 1.0,
            next_state: (),
            next_observation: (),
            done: true,
        });

        // With gamma=0.9: 1 + 0.9 + 0.81 = 2.71
        let discounted = traj.discounted_reward(0.9);
        assert!((discounted - 2.71).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate() {
        let env = GridWorld::new(3, 3);
        let policy = RandomPolicy::new(4);

        let stats = evaluate(&policy, &env, 10, 50);

        assert_eq!(stats.num_episodes, 10);
        assert!(stats.mean_length > 0.0);
    }

    #[test]
    fn test_compare_policies() {
        let env = GridWorld::new(3, 3).with_goal(2, 0);

        // Random policy
        let random = RandomPolicy::new(4);

        // Always-right policy (should perform better)
        let right = ConstantPolicy::new(3usize);

        let comparison = compare_policies(&right, &random, &env, 20, 50);

        // Right policy should generally be better
        // (This might not always be true due to randomness, but usually is)
        println!("{}", comparison);
    }
}
