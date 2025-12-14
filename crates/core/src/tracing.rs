//! # Zero-Cost Tracing (Session 6)
//!
//! This module provides compile-time configurable tracing using const generics.
//!
//! ## The Problem
//!
//! In Python agent frameworks, tracing is always enabled:
//! - Every call has timing overhead
//! - Every call allocates trace data
//! - No way to completely disable in production
//!
//! ## The Solution
//!
//! Rust's const generics let us choose tracing behavior at compile time:
//!
//! ```rust
//! use compositional_core::tracing::{Traced, TraceNode};
//!
//! // In debug builds: full tracing
//! // In release builds: zero overhead
//! ```
//!
//! ## How It Works
//!
//! - `Traced<C, false>`: No tracing code is generated. The wrapper compiles away.
//! - `Traced<C, true>`: Full tracing with timing and hierarchy.
//!
//! The choice is made at compile time via `#[cfg(debug_assertions)]`.

use std::future::Future;
use std::pin::Pin;
use std::time::{Duration, Instant};

use crate::CoreError;

/// A computation that can be executed with typed input/output.
///
/// This is the base trait for anything that can be traced.
/// Diagrams, agents, and composed pipelines all implement this.
pub trait Computation: Send + Sync {
    /// The input type for this computation.
    type Input: Send;

    /// The output type produced by this computation.
    type Output: Send;

    /// Execute the computation.
    fn run(
        &self,
        input: Self::Input,
    ) -> impl Future<Output = Result<Self::Output, CoreError>> + Send;

    /// Get the name of this computation for tracing.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

/// A node in the trace tree, recording timing and hierarchy.
#[derive(Debug, Clone)]
pub struct TraceNode {
    /// The name of the computation
    pub name: String,
    /// How long this computation took
    pub duration: Duration,
    /// Child traces (for composed computations)
    pub children: Vec<TraceNode>,
}

impl TraceNode {
    /// Create a new trace node.
    pub fn new(name: impl Into<String>, duration: Duration) -> Self {
        Self {
            name: name.into(),
            duration,
            children: Vec::new(),
        }
    }

    /// Add a child trace.
    pub fn with_child(mut self, child: TraceNode) -> Self {
        self.children.push(child);
        self
    }

    /// Add multiple children.
    pub fn with_children(mut self, children: Vec<TraceNode>) -> Self {
        self.children.extend(children);
        self
    }

    /// Pretty-print the trace tree.
    pub fn display(&self) -> String {
        self.display_indent(0)
    }

    fn display_indent(&self, indent: usize) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        let prefix = "  ".repeat(indent);
        writeln!(out, "{}[{:?}] {}", prefix, self.duration, self.name).unwrap();
        for child in &self.children {
            out.push_str(&child.display_indent(indent + 1));
        }
        out
    }

    /// Get total duration including children (for verification).
    pub fn total_duration(&self) -> Duration {
        self.duration
    }
}

impl std::fmt::Display for TraceNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display())
    }
}

/// A traced wrapper around a computation.
///
/// The const generic `ENABLED` determines whether tracing is active:
/// - `Traced<C, false>`: Zero overhead, no tracing
/// - `Traced<C, true>`: Full tracing with timing
///
/// # Example
///
/// ```rust,ignore
/// use compositional_core::tracing::{Traced, Computation};
///
/// let traced = Traced::<MyComputation, true>::new(my_computation);
/// let (result, trace) = traced.run(input).await?;
/// println!("{}", trace.display());
/// ```
#[derive(Clone)]
pub struct Traced<C, const ENABLED: bool> {
    inner: C,
}

impl<C, const ENABLED: bool> Traced<C, ENABLED> {
    /// Create a new traced wrapper.
    pub fn new(inner: C) -> Self {
        Self { inner }
    }

    /// Get a reference to the inner computation.
    pub fn inner(&self) -> &C {
        &self.inner
    }

    /// Unwrap and return the inner computation.
    pub fn into_inner(self) -> C {
        self.inner
    }
}

// ============================================================================
// Tracing Disabled: Zero Overhead
// ============================================================================

impl<C: Computation> Computation for Traced<C, false> {
    type Input = C::Input;
    type Output = C::Output; // Same output type — no trace

    fn run(
        &self,
        input: Self::Input,
    ) -> impl Future<Output = Result<Self::Output, CoreError>> + Send {
        // No tracing code generated — the compiler optimizes this away
        self.inner.run(input)
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }
}

// ============================================================================
// Tracing Enabled: Full Timing
// ============================================================================

impl<C: Computation> Computation for Traced<C, true> {
    type Input = C::Input;
    type Output = (C::Output, TraceNode); // Output includes trace

    fn run(
        &self,
        input: Self::Input,
    ) -> impl Future<Output = Result<Self::Output, CoreError>> + Send {
        let name = self.inner.name().to_string();
        let inner = &self.inner;

        async move {
            let start = Instant::now();
            let result = inner.run(input).await?;
            let trace = TraceNode::new(name, start.elapsed());
            Ok((result, trace))
        }
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }
}

// ============================================================================
// Compile-Time Selection
// ============================================================================

/// Type alias for traced computation in debug builds.
/// In debug: full tracing. In release: zero overhead.
#[cfg(debug_assertions)]
pub type AutoTraced<C> = Traced<C, true>;

/// Type alias for traced computation in release builds.
/// In debug: full tracing. In release: zero overhead.
#[cfg(not(debug_assertions))]
pub type AutoTraced<C> = Traced<C, false>;

// ============================================================================
// Helper: Run with tracing
// ============================================================================

/// A boxed future for async operations.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Extension trait for running computations with optional tracing.
pub trait ComputationExt: Computation + Sized {
    /// Wrap this computation with tracing enabled.
    fn traced(self) -> Traced<Self, true> {
        Traced::new(self)
    }

    /// Wrap this computation with tracing disabled.
    fn untraced(self) -> Traced<Self, false> {
        Traced::new(self)
    }

    /// Wrap with auto-selected tracing (debug = on, release = off).
    fn auto_traced(self) -> AutoTraced<Self> {
        Traced::new(self)
    }
}

impl<C: Computation + Sized> ComputationExt for C {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct AddOne;

    impl Computation for AddOne {
        type Input = i32;
        type Output = i32;

        async fn run(&self, input: Self::Input) -> Result<Self::Output, CoreError> {
            Ok(input + 1)
        }

        fn name(&self) -> &'static str {
            "AddOne"
        }
    }

    #[derive(Clone)]
    #[allow(dead_code)]
    struct Double;

    impl Computation for Double {
        type Input = i32;
        type Output = i32;

        async fn run(&self, input: Self::Input) -> Result<Self::Output, CoreError> {
            Ok(input * 2)
        }

        fn name(&self) -> &'static str {
            "Double"
        }
    }

    #[derive(Clone)]
    struct SlowComputation {
        delay_ms: u64,
    }

    impl Computation for SlowComputation {
        type Input = ();
        type Output = ();

        async fn run(&self, _input: Self::Input) -> Result<Self::Output, CoreError> {
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            Ok(())
        }

        fn name(&self) -> &'static str {
            "SlowComputation"
        }
    }

    #[tokio::test]
    async fn test_traced_disabled_same_output() {
        let comp = AddOne;
        let traced: Traced<AddOne, false> = Traced::new(comp);

        let result = traced.run(5).await.unwrap();
        assert_eq!(result, 6); // Just the output, no trace
    }

    #[tokio::test]
    async fn test_traced_enabled_includes_trace() {
        let comp = AddOne;
        let traced: Traced<AddOne, true> = Traced::new(comp);

        let (result, trace) = traced.run(5).await.unwrap();
        assert_eq!(result, 6);
        assert_eq!(trace.name, "AddOne");
        // Note: duration may be 0 for very fast operations on some systems
        // Timing accuracy is tested in test_trace_timing_accuracy
    }

    #[tokio::test]
    async fn test_trace_timing_accuracy() {
        let comp = SlowComputation { delay_ms: 50 };
        let traced: Traced<SlowComputation, true> = Traced::new(comp);

        let ((), trace) = traced.run(()).await.unwrap();

        // Should be at least 50ms (allow some tolerance)
        assert!(trace.duration.as_millis() >= 45);
        assert!(trace.duration.as_millis() < 200); // But not too slow
    }

    #[test]
    fn test_trace_node_display() {
        let trace = TraceNode::new("Parent", Duration::from_millis(100))
            .with_child(TraceNode::new("Child1", Duration::from_millis(30)))
            .with_child(TraceNode::new("Child2", Duration::from_millis(40)));

        let display = trace.display();
        assert!(display.contains("Parent"));
        assert!(display.contains("Child1"));
        assert!(display.contains("Child2"));
        assert!(display.contains("100ms"));
    }

    #[test]
    fn test_traced_wrapper_accessors() {
        let comp = AddOne;
        let traced = Traced::<AddOne, true>::new(comp);

        assert_eq!(traced.inner().name(), "AddOne");

        let inner = traced.into_inner();
        assert_eq!(inner.name(), "AddOne");
    }

    #[tokio::test]
    async fn test_computation_ext_traced() {
        let result = AddOne.traced().run(10).await.unwrap();
        assert_eq!(result.0, 11);
        assert_eq!(result.1.name, "AddOne");
    }

    #[tokio::test]
    async fn test_computation_ext_untraced() {
        let result = AddOne.untraced().run(10).await.unwrap();
        assert_eq!(result, 11);
    }

    #[test]
    fn test_trace_node_with_children() {
        let child1 = TraceNode::new("op1", Duration::from_millis(10));
        let child2 = TraceNode::new("op2", Duration::from_millis(20));

        let parent = TraceNode::new("pipeline", Duration::from_millis(50))
            .with_children(vec![child1, child2]);

        assert_eq!(parent.children.len(), 2);
        assert_eq!(parent.children[0].name, "op1");
        assert_eq!(parent.children[1].name, "op2");
    }
}
