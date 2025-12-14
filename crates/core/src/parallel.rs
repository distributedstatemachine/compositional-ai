//! # Parallel Agents - Runtime Tensor Product (Session 5)
//!
//! This module provides the runtime interpretation of the tensor product (⊗).
//! While [`crate::diagram::Diagram::tensor`] represents parallel structure *statically*,
//! `ParallelAgents` *executes* agents concurrently.
//!
//! ## Key Concepts
//!
//! - **Agent**: An async computation with typed input/output
//! - **ParallelAgents**: A collection of agents that run concurrently
//! - **Combiner**: Merges results from parallel agents (fan-in)
//!
//! ## Pattern: Fan-out → Process → Fan-in
//!
//! ```text
//!               ┌─────────────────┐
//!    query ─────│ WebSearch(tech) │─────┐
//!               └─────────────────┘     │    ┌───────────┐
//!                                       ├────│ Synthesis │──── report
//!               ┌─────────────────┐     │    └───────────┘
//!    query ─────│ WebSearch(acad) │─────┘
//!               └─────────────────┘
//!
//!    Diagram: (search_tech ⊗ search_acad) ; synthesize
//! ```
//!
//! ## Correspondence with Diagrams
//!
//! | Static (Diagram) | Dynamic (Runtime) |
//! |------------------|-------------------|
//! | `f.tensor(g)` | `ParallelAgents::new().add(f).add(g)` |
//! | Side-by-side structure | Concurrent execution |
//! | `f.then(g)` | Sequential `await` |

use std::future::Future;
use std::hash::Hash;
use std::pin::Pin;

/// The Agent trait defines a minimal interface for parallel execution.
///
/// An agent is an async computation that takes an input and produces an output.
/// Agents must be cloneable to allow spawning multiple copies for parallel execution.
///
/// # Example
///
/// ```rust
/// use compositional_core::parallel::Agent;
/// use std::future::Future;
///
/// #[derive(Clone)]
/// struct Echo;
///
/// impl Agent for Echo {
///     type Input = String;
///     type Output = String;
///     type Error = std::convert::Infallible;
///
///     fn run(&self, input: Self::Input) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
///         async move { Ok(input) }
///     }
/// }
/// ```
pub trait Agent: Clone + Send + Sync + 'static {
    /// The input type for this agent.
    type Input: Send + 'static;

    /// The output type produced by this agent.
    type Output: Send + 'static;

    /// The error type that may be returned.
    type Error: Send + 'static;

    /// Execute the agent with the given input.
    fn run(
        &self,
        input: Self::Input,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send;
}

/// A boxed future for dynamic dispatch of agent results.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// A parallel executor that runs multiple agents concurrently.
///
/// This is the runtime interpretation of the tensor product (⊗).
/// Where `Diagram::tensor` represents parallel structure statically,
/// `ParallelAgents` executes agents concurrently at runtime.
///
/// # Example
///
/// ```rust
/// use compositional_core::parallel::{Agent, ParallelAgents};
///
/// #[derive(Clone)]
/// struct DoubleAgent;
///
/// impl Agent for DoubleAgent {
///     type Input = i32;
///     type Output = i32;
///     type Error = std::convert::Infallible;
///
///     fn run(&self, input: Self::Input) -> impl std::future::Future<Output = Result<Self::Output, Self::Error>> + Send {
///         async move { Ok(input * 2) }
///     }
/// }
///
/// # #[tokio::main]
/// # async fn main() {
/// let agents = ParallelAgents::new()
///     .with(DoubleAgent)
///     .with(DoubleAgent);
///
/// // Each agent gets its own input
/// let results = agents.run_all(vec![1, 2]).await.unwrap();
/// assert_eq!(results.len(), 2);
/// # }
/// ```
#[derive(Clone)]
pub struct ParallelAgents<A> {
    agents: Vec<A>,
}

impl<A> ParallelAgents<A> {
    /// Create a new empty parallel agent group.
    pub fn new() -> Self {
        Self { agents: Vec::new() }
    }

    /// Add an agent to the group.
    pub fn with(mut self, agent: A) -> Self {
        self.agents.push(agent);
        self
    }

    /// Get the number of agents in this group.
    pub fn len(&self) -> usize {
        self.agents.len()
    }

    /// Check if this group is empty.
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    /// Tensor two parallel groups together.
    ///
    /// This is the monoidal product for parallel agent groups:
    /// `(a₁ ⊗ a₂) ⊗ (b₁ ⊗ b₂) = a₁ ⊗ a₂ ⊗ b₁ ⊗ b₂`
    pub fn tensor(mut self, mut other: Self) -> Self {
        self.agents.append(&mut other.agents);
        self
    }
}

impl<A> Default for ParallelAgents<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A> ParallelAgents<A>
where
    A: Agent,
{
    /// Execute all agents in parallel with the provided inputs.
    ///
    /// Each agent receives its corresponding input from the vector.
    /// Results are returned in the order they complete (not input order).
    ///
    /// # Panics
    ///
    /// Panics if the number of inputs doesn't match the number of agents.
    pub async fn run_all(&self, inputs: Vec<A::Input>) -> Result<Vec<A::Output>, A::Error> {
        assert_eq!(
            inputs.len(),
            self.agents.len(),
            "Input count must match agent count"
        );

        let mut handles = Vec::with_capacity(self.agents.len());

        // Spawn all agents concurrently
        for (agent, input) in self.agents.iter().zip(inputs) {
            let agent = agent.clone();
            let handle = tokio::spawn(async move { agent.run(input).await });
            handles.push(handle);
        }

        // Collect results
        let mut results = Vec::with_capacity(handles.len());
        for handle in handles {
            let result = handle.await.expect("Agent task panicked")?;
            results.push(result);
        }

        Ok(results)
    }

    /// Fan-out: broadcast the same input to all agents.
    ///
    /// This is useful when you want multiple agents to process the same data
    /// and then combine their results.
    pub async fn fan_out(&self, input: A::Input) -> Result<Vec<A::Output>, A::Error>
    where
        A::Input: Clone,
    {
        let inputs = vec![input; self.agents.len()];
        self.run_all(inputs).await
    }
}

// ============================================================================
// Combiners: Fan-in patterns for merging parallel results
// ============================================================================

/// A combiner merges results from parallel agents.
///
/// This completes the parallel pattern: fan-out → process → fan-in.
pub trait Combiner<T> {
    /// The output type after combining.
    type Output;

    /// Combine multiple results into a single output.
    fn combine(&self, results: Vec<T>) -> Self::Output;
}

/// Concatenate string results with newlines.
#[derive(Debug, Clone, Copy, Default)]
pub struct Concat;

impl Combiner<String> for Concat {
    type Output = String;

    fn combine(&self, results: Vec<String>) -> String {
        results.join("\n")
    }
}

/// Concatenate with a custom separator.
#[derive(Debug, Clone)]
pub struct ConcatWith {
    separator: String,
}

impl ConcatWith {
    /// Create a new combiner with the given separator.
    pub fn new(separator: impl Into<String>) -> Self {
        Self {
            separator: separator.into(),
        }
    }
}

impl Combiner<String> for ConcatWith {
    type Output = String;

    fn combine(&self, results: Vec<String>) -> String {
        results.join(&self.separator)
    }
}

/// Majority voting combiner.
///
/// Returns the most common result, or `None` if empty.
#[derive(Debug, Clone, Copy, Default)]
pub struct Vote;

impl<T: Eq + Hash + Clone> Combiner<T> for Vote {
    type Output = Option<T>;

    fn combine(&self, results: Vec<T>) -> Option<T> {
        use std::collections::HashMap;

        if results.is_empty() {
            return None;
        }

        let mut counts: HashMap<T, usize> = HashMap::new();
        for r in results {
            *counts.entry(r).or_insert(0) += 1;
        }

        counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(value, _)| value)
    }
}

/// Collect all results into a vector (identity combiner).
#[derive(Debug, Clone, Copy, Default)]
pub struct Collect;

impl<T> Combiner<T> for Collect {
    type Output = Vec<T>;

    fn combine(&self, results: Vec<T>) -> Vec<T> {
        results
    }
}

/// Take the first result, ignoring the rest.
#[derive(Debug, Clone, Copy, Default)]
pub struct First;

impl<T> Combiner<T> for First {
    type Output = Option<T>;

    fn combine(&self, mut results: Vec<T>) -> Option<T> {
        if results.is_empty() {
            None
        } else {
            Some(results.swap_remove(0))
        }
    }
}

/// Combine results using a custom function.
pub struct FnCombiner<F> {
    f: F,
}

impl<F> FnCombiner<F> {
    /// Create a combiner from a function.
    pub fn new(f: F) -> Self {
        Self { f }
    }
}

impl<T, O, F> Combiner<T> for FnCombiner<F>
where
    F: Fn(Vec<T>) -> O,
{
    type Output = O;

    fn combine(&self, results: Vec<T>) -> O {
        (self.f)(results)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct DoubleAgent;

    impl Agent for DoubleAgent {
        type Input = i32;
        type Output = i32;
        type Error = std::convert::Infallible;

        fn run(
            &self,
            input: Self::Input,
        ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
            async move { Ok(input * 2) }
        }
    }

    #[derive(Clone)]
    struct StringAgent {
        prefix: String,
    }

    impl Agent for StringAgent {
        type Input = String;
        type Output = String;
        type Error = std::convert::Infallible;

        fn run(
            &self,
            input: Self::Input,
        ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
            let prefix = self.prefix.clone();
            async move { Ok(format!("{}: {}", prefix, input)) }
        }
    }

    #[tokio::test]
    async fn test_parallel_agents_run_all() {
        let agents = ParallelAgents::new()
            .with(DoubleAgent)
            .with(DoubleAgent)
            .with(DoubleAgent);

        let results = agents.run_all(vec![1, 2, 3]).await.unwrap();

        assert_eq!(results.len(), 3);
        // Results should contain doubled values
        assert!(results.contains(&2));
        assert!(results.contains(&4));
        assert!(results.contains(&6));
    }

    #[tokio::test]
    async fn test_parallel_agents_fan_out() {
        let agents = ParallelAgents::new()
            .with(StringAgent {
                prefix: "A".to_string(),
            })
            .with(StringAgent {
                prefix: "B".to_string(),
            });

        let results = agents.fan_out("hello".to_string()).await.unwrap();

        assert_eq!(results.len(), 2);
        assert!(results.contains(&"A: hello".to_string()));
        assert!(results.contains(&"B: hello".to_string()));
    }

    #[tokio::test]
    async fn test_parallel_agents_tensor() {
        let group1 = ParallelAgents::new().with(DoubleAgent).with(DoubleAgent);

        let group2 = ParallelAgents::new().with(DoubleAgent);

        let combined = group1.tensor(group2);

        assert_eq!(combined.len(), 3);

        let results = combined.run_all(vec![1, 2, 3]).await.unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_combiner_concat() {
        let combiner = Concat;
        let results = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert_eq!(combiner.combine(results), "a\nb\nc");
    }

    #[test]
    fn test_combiner_concat_with() {
        let combiner = ConcatWith::new(", ");
        let results = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert_eq!(combiner.combine(results), "a, b, c");
    }

    #[test]
    fn test_combiner_vote() {
        let combiner = Vote;

        // Clear majority
        let results = vec![1, 2, 1, 1, 3];
        assert_eq!(combiner.combine(results), Some(1));

        // Empty case
        let empty: Vec<i32> = vec![];
        assert_eq!(combiner.combine(empty), None);
    }

    #[test]
    fn test_combiner_collect() {
        let combiner = Collect;
        let results = vec![1, 2, 3];
        assert_eq!(combiner.combine(results), vec![1, 2, 3]);
    }

    #[test]
    fn test_combiner_first() {
        let combiner = First;

        let results = vec![1, 2, 3];
        assert_eq!(combiner.combine(results), Some(1));

        let empty: Vec<i32> = vec![];
        assert_eq!(combiner.combine(empty), None);
    }

    #[test]
    fn test_combiner_fn() {
        let combiner = FnCombiner::new(|v: Vec<i32>| v.iter().sum::<i32>());
        let results = vec![1, 2, 3, 4];
        assert_eq!(combiner.combine(results), 10);
    }

    #[test]
    fn test_parallel_agents_empty() {
        let agents: ParallelAgents<DoubleAgent> = ParallelAgents::new();
        assert!(agents.is_empty());
        assert_eq!(agents.len(), 0);
    }

    #[test]
    fn test_parallel_agents_with() {
        let agents = ParallelAgents::new().with(DoubleAgent).with(DoubleAgent);

        assert!(!agents.is_empty());
        assert_eq!(agents.len(), 2);
    }
}
