//! # Compositional Games (Session 17)
//!
//! This crate implements **Open Games** - a compositional framework for game theory where:
//! - Games are modular components (boxes) with typed interfaces
//! - Complex games are built by composing simpler ones
//! - Policy, environment, and objective can be swapped independently
//!
//! ## Core Components
//!
//! - [`Policy`]: Maps observations to actions (`Obs -> Act`)
//! - [`Env`]: Defines world dynamics (`(State, Act) -> (State, Obs, Reward)`)
//! - [`rollout()`]: Executes a policy in an environment for T steps
//!
//! ## Example
//!
//! ```rust
//! use compositional_games::{Policy, Env, Trajectory, rollout};
//! use compositional_games::env::GridWorld;
//! use compositional_games::policy::RandomPolicy;
//!
//! // Create environment and policy
//! let env = GridWorld::new(5, 5);
//! let policy = RandomPolicy::new(4); // 4 actions: up, down, left, right
//!
//! // Run rollout
//! let trajectory = rollout(&policy, &env, 10);
//! println!("Total reward: {}", trajectory.total_reward());
//! ```
//!
//! ## Compositional Design
//!
//! The key insight is that policies and environments are **swappable**:
//!
//! ```text
//!          ┌──────────┐
//!    Obs ─▶│  Policy  │─▶ Act
//!          └──────────┘
//!               │
//!               ▼
//!          ┌──────────┐
//!          │   Env    │─▶ (State, Obs, Reward)
//!          └──────────┘
//! ```
//!
//! Swap the policy → different decision rule, same environment
//! Swap the environment → same policy, different world dynamics

pub mod env;
mod error;
pub mod policy;
pub mod rollout;

pub use env::Env;
pub use error::GameError;
pub use policy::Policy;
pub use rollout::{
    compare_policies, evaluate, rollout, PolicyComparison, RolloutStats, Step, Trajectory,
};
