//! Environments: World dynamics and state transitions.
//!
//! An environment defines how the world evolves in response to actions.
//! This is another swappable component in open games.
//!
//! # Compositional Design
//!
//! Environments are morphisms: `(State, Act) → (State, Obs, Reward)`
//!
//! ```text
//!        ┌──────────┐
//!  Act ─▶│   Env    │─▶ (State', Obs, Reward)
//!        │          │
//! State ─▶│          │
//!        └──────────┘
//! ```
//!
//! You can swap environments without changing the policy!

use crate::GameError;

/// Environment transition result.
#[derive(Debug, Clone)]
pub struct Transition<State, Obs> {
    /// New state after action
    pub next_state: State,
    /// Observation from the new state
    pub observation: Obs,
    /// Reward received
    pub reward: f64,
    /// Whether the episode has ended
    pub done: bool,
}

/// An environment defines world dynamics.
///
/// This is a key swappable component: same policy, different environment
/// = different world to operate in.
pub trait Env {
    /// State type
    type State: Clone;
    /// Observation type (what the agent sees)
    type Obs;
    /// Action type
    type Act;

    /// Get the initial state.
    fn initial_state(&self) -> Self::State;

    /// Get observation from a state.
    fn observe(&self, state: &Self::State) -> Self::Obs;

    /// Take a step: (state, action) → (next_state, obs, reward, done)
    fn step(
        &self,
        state: &Self::State,
        action: &Self::Act,
    ) -> Result<Transition<Self::State, Self::Obs>, GameError>;

    /// Check if state is terminal.
    fn is_terminal(&self, state: &Self::State) -> bool;

    /// Get available actions in a state (optional).
    fn available_actions(&self, _state: &Self::State) -> Option<Vec<Self::Act>> {
        None
    }

    /// Reset to initial state and return observation.
    fn reset(&self) -> (Self::State, Self::Obs) {
        let state = self.initial_state();
        let obs = self.observe(&state);
        (state, obs)
    }
}

// ============================================================================
// Grid World Environment
// ============================================================================

/// A simple grid world environment.
///
/// Agent starts at (0, 0), goal is at (width-1, height-1).
/// Actions: 0=up, 1=down, 2=left, 3=right
#[derive(Debug, Clone)]
pub struct GridWorld {
    /// Grid width
    pub width: usize,
    /// Grid height
    pub height: usize,
    /// Goal position
    pub goal: (usize, usize),
    /// Obstacles (positions that can't be entered)
    pub obstacles: Vec<(usize, usize)>,
    /// Step penalty
    pub step_reward: f64,
    /// Goal reward
    pub goal_reward: f64,
}

/// Grid world state: agent position
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridState {
    pub x: usize,
    pub y: usize,
}

/// Grid world observation (same as state for fully observable)
pub type GridObs = GridState;

impl GridWorld {
    /// Create a new grid world.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            goal: (width - 1, height - 1),
            obstacles: Vec::new(),
            step_reward: -0.1,
            goal_reward: 10.0,
        }
    }

    /// Set the goal position.
    pub fn with_goal(mut self, x: usize, y: usize) -> Self {
        self.goal = (x, y);
        self
    }

    /// Add obstacles.
    pub fn with_obstacles(mut self, obstacles: Vec<(usize, usize)>) -> Self {
        self.obstacles = obstacles;
        self
    }

    /// Set rewards.
    pub fn with_rewards(mut self, step: f64, goal: f64) -> Self {
        self.step_reward = step;
        self.goal_reward = goal;
        self
    }

    /// Check if a position is valid (in bounds and not obstacle).
    fn is_valid(&self, x: usize, y: usize) -> bool {
        x < self.width && y < self.height && !self.obstacles.contains(&(x, y))
    }
}

impl Env for GridWorld {
    type State = GridState;
    type Obs = GridObs;
    type Act = usize;

    fn initial_state(&self) -> Self::State {
        GridState { x: 0, y: 0 }
    }

    fn observe(&self, state: &Self::State) -> Self::Obs {
        *state
    }

    fn step(
        &self,
        state: &Self::State,
        action: &Self::Act,
    ) -> Result<Transition<Self::State, Self::Obs>, GameError> {
        if self.is_terminal(state) {
            return Err(GameError::TerminalState);
        }

        // Compute new position based on action
        let (new_x, new_y) = match action {
            0 => (state.x, state.y.saturating_add(1)), // up
            1 => (state.x, state.y.saturating_sub(1)), // down
            2 => (state.x.saturating_sub(1), state.y), // left
            3 => (state.x.saturating_add(1), state.y), // right
            _ => return Err(GameError::InvalidAction { action: *action }),
        };

        // Check bounds and obstacles
        let (final_x, final_y) = if self.is_valid(new_x, new_y) {
            (new_x, new_y)
        } else {
            (state.x, state.y) // Stay in place if invalid move
        };

        let next_state = GridState {
            x: final_x,
            y: final_y,
        };

        // Check if goal reached
        let at_goal = (final_x, final_y) == self.goal;
        let reward = if at_goal {
            self.goal_reward
        } else {
            self.step_reward
        };

        Ok(Transition {
            next_state,
            observation: next_state,
            reward,
            done: at_goal,
        })
    }

    fn is_terminal(&self, state: &Self::State) -> bool {
        (state.x, state.y) == self.goal
    }

    fn available_actions(&self, _state: &Self::State) -> Option<Vec<Self::Act>> {
        Some(vec![0, 1, 2, 3])
    }
}

// ============================================================================
// Bandit Environment
// ============================================================================

/// A multi-armed bandit environment.
///
/// Each arm has a different reward distribution.
#[derive(Debug, Clone)]
pub struct Bandit {
    /// Mean rewards for each arm
    pub means: Vec<f64>,
    /// Standard deviations for each arm
    pub stds: Vec<f64>,
    /// Random seed
    seed: std::cell::Cell<u64>,
}

impl Bandit {
    /// Create a new bandit with given mean rewards.
    pub fn new(means: Vec<f64>) -> Self {
        let n = means.len();
        Self {
            means,
            stds: vec![1.0; n],
            seed: std::cell::Cell::new(42),
        }
    }

    /// Set standard deviations.
    pub fn with_stds(mut self, stds: Vec<f64>) -> Self {
        self.stds = stds;
        self
    }

    /// Simple Gaussian random number (Box-Muller).
    fn sample_gaussian(&self, mean: f64, std: f64) -> f64 {
        let s = self.seed.get();
        let s1 = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let s2 = s1.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.seed.set(s2);

        let u1 = (s1 >> 33) as f64 / (1u64 << 31) as f64;
        let u2 = (s2 >> 33) as f64 / (1u64 << 31) as f64;

        let u1 = u1.max(1e-10); // Avoid log(0)
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

        mean + std * z
    }
}

impl Env for Bandit {
    type State = ();
    type Obs = ();
    type Act = usize;

    fn initial_state(&self) -> Self::State {}

    fn observe(&self, _state: &Self::State) -> Self::Obs {}

    fn step(
        &self,
        _state: &Self::State,
        action: &Self::Act,
    ) -> Result<Transition<Self::State, Self::Obs>, GameError> {
        if *action >= self.means.len() {
            return Err(GameError::InvalidAction { action: *action });
        }

        let reward = self.sample_gaussian(self.means[*action], self.stds[*action]);

        Ok(Transition {
            next_state: (),
            observation: (),
            reward,
            done: false, // Bandits never terminate
        })
    }

    fn is_terminal(&self, _state: &Self::State) -> bool {
        false
    }

    fn available_actions(&self, _state: &Self::State) -> Option<Vec<Self::Act>> {
        Some((0..self.means.len()).collect())
    }
}

// ============================================================================
// Chain MDP
// ============================================================================

/// A simple chain MDP for testing.
///
/// States: 0, 1, 2, ..., n-1
/// Actions: 0=left, 1=right
/// Reward: +1 at state n-1, 0 otherwise
#[derive(Debug, Clone)]
pub struct ChainMDP {
    /// Number of states
    pub num_states: usize,
    /// Slip probability (probability of moving opposite direction)
    pub slip_prob: f64,
    /// Random seed
    seed: std::cell::Cell<u64>,
}

impl ChainMDP {
    /// Create a new chain MDP.
    pub fn new(num_states: usize) -> Self {
        Self {
            num_states,
            slip_prob: 0.0,
            seed: std::cell::Cell::new(42),
        }
    }

    /// Set slip probability.
    pub fn with_slip(mut self, slip_prob: f64) -> Self {
        self.slip_prob = slip_prob.clamp(0.0, 1.0);
        self
    }

    fn next_random(&self) -> f64 {
        let s = self.seed.get();
        let new_seed = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.seed.set(new_seed);
        (new_seed >> 33) as f64 / (1u64 << 31) as f64
    }
}

impl Env for ChainMDP {
    type State = usize;
    type Obs = usize;
    type Act = usize;

    fn initial_state(&self) -> Self::State {
        0
    }

    fn observe(&self, state: &Self::State) -> Self::Obs {
        *state
    }

    fn step(
        &self,
        state: &Self::State,
        action: &Self::Act,
    ) -> Result<Transition<Self::State, Self::Obs>, GameError> {
        if *action > 1 {
            return Err(GameError::InvalidAction { action: *action });
        }

        // Determine actual movement (with possible slip)
        let actual_action = if self.next_random() < self.slip_prob {
            1 - action // Slip: opposite direction
        } else {
            *action
        };

        let next_state = if actual_action == 0 {
            state.saturating_sub(1) // left
        } else {
            (*state + 1).min(self.num_states - 1) // right
        };

        let done = next_state == self.num_states - 1;
        let reward = if done { 1.0 } else { 0.0 };

        Ok(Transition {
            next_state,
            observation: next_state,
            reward,
            done,
        })
    }

    fn is_terminal(&self, state: &Self::State) -> bool {
        *state == self.num_states - 1
    }

    fn available_actions(&self, _state: &Self::State) -> Option<Vec<Self::Act>> {
        Some(vec![0, 1])
    }
}

// ============================================================================
// Function Environment
// ============================================================================

/// An environment defined by functions.
///
/// Allows creating custom environments without implementing the trait.
pub struct FnEnv<State, Obs, Act, I, O, S, T> {
    /// Initial state function
    pub init: I,
    /// Observation function
    pub obs: O,
    /// Step function
    pub step_fn: S,
    /// Terminal check function
    pub terminal: T,
    _phantom: std::marker::PhantomData<(State, Obs, Act)>,
}

impl<State, Obs, Act, I, O, S, T> FnEnv<State, Obs, Act, I, O, S, T>
where
    State: Clone,
    I: Fn() -> State,
    O: Fn(&State) -> Obs,
    S: Fn(&State, &Act) -> Result<Transition<State, Obs>, GameError>,
    T: Fn(&State) -> bool,
{
    /// Create a new function-based environment.
    pub fn new(init: I, obs: O, step_fn: S, terminal: T) -> Self {
        Self {
            init,
            obs,
            step_fn,
            terminal,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<State, Obs, Act, I, O, S, T> Env for FnEnv<State, Obs, Act, I, O, S, T>
where
    State: Clone,
    I: Fn() -> State,
    O: Fn(&State) -> Obs,
    S: Fn(&State, &Act) -> Result<Transition<State, Obs>, GameError>,
    T: Fn(&State) -> bool,
{
    type State = State;
    type Obs = Obs;
    type Act = Act;

    fn initial_state(&self) -> Self::State {
        (self.init)()
    }

    fn observe(&self, state: &Self::State) -> Self::Obs {
        (self.obs)(state)
    }

    fn step(
        &self,
        state: &Self::State,
        action: &Self::Act,
    ) -> Result<Transition<Self::State, Self::Obs>, GameError> {
        (self.step_fn)(state, action)
    }

    fn is_terminal(&self, state: &Self::State) -> bool {
        (self.terminal)(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_world_basic() {
        let env = GridWorld::new(5, 5);
        let (state, obs) = env.reset();

        assert_eq!(state.x, 0);
        assert_eq!(state.y, 0);
        assert_eq!(obs, state);
    }

    #[test]
    fn test_grid_world_movement() {
        let env = GridWorld::new(5, 5);
        let state = GridState { x: 2, y: 2 };

        // Move right
        let trans = env.step(&state, &3).unwrap();
        assert_eq!(trans.next_state.x, 3);
        assert_eq!(trans.next_state.y, 2);

        // Move up
        let trans = env.step(&state, &0).unwrap();
        assert_eq!(trans.next_state.x, 2);
        assert_eq!(trans.next_state.y, 3);
    }

    #[test]
    fn test_grid_world_goal() {
        let env = GridWorld::new(3, 3);
        let state = GridState { x: 1, y: 2 };

        // Move to goal (2, 2)
        let trans = env.step(&state, &3).unwrap();
        assert!(trans.done);
        assert!(trans.reward > 0.0);
    }

    #[test]
    fn test_bandit() {
        let env = Bandit::new(vec![0.0, 1.0, 2.0]);
        let state = env.initial_state();

        // Pull arms multiple times
        let mut rewards = vec![0.0; 3];
        for arm in 0..3 {
            for _ in 0..100 {
                let trans = env.step(&state, &arm).unwrap();
                rewards[arm] += trans.reward;
            }
            rewards[arm] /= 100.0;
        }

        // Arm 2 should have highest average (around 2.0)
        assert!(rewards[2] > rewards[1]);
        assert!(rewards[1] > rewards[0]);
    }

    #[test]
    fn test_chain_mdp() {
        let env = ChainMDP::new(5);
        let mut state = env.initial_state();

        // Move right until terminal
        while !env.is_terminal(&state) {
            let trans = env.step(&state, &1).unwrap();
            state = trans.next_state;
        }

        assert_eq!(state, 4);
    }
}
