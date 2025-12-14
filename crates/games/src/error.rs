//! Error types for game operations.

use thiserror::Error;

/// Errors that can occur in game operations.
#[derive(Debug, Clone, Error)]
pub enum GameError {
    /// Invalid action for the current state.
    #[error("Invalid action {action} for state")]
    InvalidAction { action: usize },

    /// Environment is in a terminal state.
    #[error("Environment is in terminal state")]
    TerminalState,

    /// State out of bounds.
    #[error("State out of bounds: {state}")]
    OutOfBounds { state: String },

    /// Policy error.
    #[error("Policy error: {message}")]
    PolicyError { message: String },
}
