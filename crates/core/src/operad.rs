//! # Operads — Multi-Input Wiring Constraints (Session 19)
//!
//! Operads enforce **arity constraints** that plain diagrams don't catch.
//! While diagrams allow any wiring if types match, operads ensure that
//! operations receive exactly the right number of inputs.
//!
//! ## Key Insight
//!
//! Diagrams: nodes + edges, any wiring allowed if types match.
//! Operads: explicit **nesting structure** — which operations can contain which.
//!
//! ## What Operads Catch That Diagrams Don't
//!
//! 1. **Arity mismatches** — wrong number of inputs
//! 2. **Nesting violations** — operation A can't be inside operation B
//! 3. **Scope constraints** — variable X only visible within block Y
//!
//! ## Motivating Example: LLM Tool-Use Pipeline
//!
//! ```text
//! Agent(3) — takes 3 tool outputs, produces 1 response
//!   ├── Tool("search")    — 1 query  → 1 result
//!   ├── Tool("calculate") — 2 numbers → 1 result
//!   └── Tool("fetch")     — 1 url    → 1 result
//!
//! Valid wiring:   Agent receives (search_result, calc_result, fetch_result)
//! Invalid wiring: Agent receives (search_result, search_result) — wrong arity!
//! ```
//!
//! With plain `Diagram`: you could accidentally wire 2 inputs instead of 3.
//! With `Operad`: the operation signature `Agent(3)` enforces exactly 3 inputs.
//!
//! ## Example
//!
//! ```rust
//! use compositional_core::operad::{Operation, WiringPlan};
//! use compositional_core::Shape;
//!
//! // Define an agent that expects 3 tool outputs
//! let agent = Operation::new("ReasoningAgent", 3)
//!     .with_inputs(vec![Shape::f32_vector(100); 3])
//!     .with_outputs(vec![Shape::f32_vector(100)]);
//!
//! // Define tools
//! let search = Operation::new("Search", 0)
//!     .with_outputs(vec![Shape::f32_vector(100)]);
//! let calc = Operation::new("Calculator", 0)
//!     .with_outputs(vec![Shape::f32_vector(100)]);
//! let fetch = Operation::new("WebFetch", 0)
//!     .with_outputs(vec![Shape::f32_vector(100)]);
//!
//! // Create a valid wiring plan
//! let plan = WiringPlan::new(agent)
//!     .with_inner(vec![search, calc, fetch])
//!     .with_wiring(vec![(0, 0), (1, 1), (2, 2)]);
//!
//! assert!(plan.validate().is_ok());  // Correct arity!
//! ```

use crate::diagram::{Diagram, Node, Port};
use crate::error::CoreError;
use crate::shape::Shape;
use std::collections::HashSet;
use std::fmt;

// ============================================================================
// OperadError
// ============================================================================

/// Errors specific to operadic composition.
#[derive(Debug, Clone, PartialEq)]
pub enum OperadError {
    /// Wrong number of inputs for operation.
    ArityMismatch {
        operation: String,
        expected: usize,
        got: usize,
    },
    /// Input/output shapes don't match.
    ShapeMismatch {
        slot: usize,
        expected: Shape,
        got: Shape,
    },
    /// Same slot wired multiple times.
    DuplicateWiring { slot: usize },
    /// Slot not wired.
    UnwiredSlot { slot: usize },
    /// Referenced operation doesn't exist.
    InvalidOperation { index: usize },
    /// Invalid slot index.
    InvalidSlot { index: usize },
    /// Wiring references non-existent inner operation.
    InvalidWiringSource { inner_idx: usize, max: usize },
    /// Wiring references non-existent slot.
    InvalidWiringTarget { slot_idx: usize, max: usize },
}

impl fmt::Display for OperadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OperadError::ArityMismatch {
                operation,
                expected,
                got,
            } => {
                write!(
                    f,
                    "Arity mismatch for '{}': expected {} inputs, got {}",
                    operation, expected, got
                )
            }
            OperadError::ShapeMismatch {
                slot,
                expected,
                got,
            } => {
                write!(
                    f,
                    "Shape mismatch at slot {}: expected {}, got {}",
                    slot, expected, got
                )
            }
            OperadError::DuplicateWiring { slot } => {
                write!(f, "Slot {} is wired multiple times", slot)
            }
            OperadError::UnwiredSlot { slot } => {
                write!(f, "Slot {} is not wired", slot)
            }
            OperadError::InvalidOperation { index } => {
                write!(f, "Invalid operation index: {}", index)
            }
            OperadError::InvalidSlot { index } => {
                write!(f, "Invalid slot index: {}", index)
            }
            OperadError::InvalidWiringSource { inner_idx, max } => {
                write!(
                    f,
                    "Invalid wiring source: inner index {} but only {} inner operations",
                    inner_idx, max
                )
            }
            OperadError::InvalidWiringTarget { slot_idx, max } => {
                write!(
                    f,
                    "Invalid wiring target: slot {} but outer has {} inputs",
                    slot_idx, max
                )
            }
        }
    }
}

impl std::error::Error for OperadError {}

// ============================================================================
// Operation
// ============================================================================

/// An operation with explicit arity.
///
/// Unlike a plain diagram node, an `Operation` has a fixed number of expected
/// inputs that must be satisfied during composition.
///
/// # Example
///
/// ```rust
/// use compositional_core::operad::Operation;
/// use compositional_core::Shape;
///
/// // A tool that takes no external inputs
/// let search_tool = Operation::new("SearchTool", 0)
///     .with_outputs(vec![Shape::f32_vector(256)]);
///
/// // An agent that requires 3 tool outputs
/// let agent = Operation::new("Agent", 3)
///     .with_inputs(vec![
///         Shape::f32_vector(256),
///         Shape::f32_vector(256),
///         Shape::f32_vector(256),
///     ])
///     .with_outputs(vec![Shape::f32_vector(512)]);
///
/// assert_eq!(agent.arity, 3);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Operation {
    /// Human-readable name.
    pub name: String,
    /// Number of inputs expected (arity).
    pub arity: usize,
    /// Input shapes (must have length == arity).
    pub inputs: Vec<Shape>,
    /// Output shapes.
    pub outputs: Vec<Shape>,
}

impl Operation {
    /// Create a new operation with given name and arity.
    pub fn new(name: impl Into<String>, arity: usize) -> Self {
        Self {
            name: name.into(),
            arity,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Set input shapes.
    ///
    /// # Panics
    ///
    /// Panics if `inputs.len() != self.arity`.
    pub fn with_inputs(mut self, inputs: Vec<Shape>) -> Self {
        assert_eq!(
            inputs.len(),
            self.arity,
            "inputs.len() ({}) must equal arity ({})",
            inputs.len(),
            self.arity
        );
        self.inputs = inputs;
        self
    }

    /// Set output shapes.
    pub fn with_outputs(mut self, outputs: Vec<Shape>) -> Self {
        self.outputs = outputs;
        self
    }

    /// Get the first output shape, if any.
    pub fn output_shape(&self) -> Option<&Shape> {
        self.outputs.first()
    }

    /// Get input shape at given slot.
    pub fn input_shape(&self, slot: usize) -> Option<&Shape> {
        self.inputs.get(slot)
    }

    /// Check if this operation has all its shapes defined.
    pub fn is_fully_typed(&self) -> bool {
        self.inputs.len() == self.arity && !self.outputs.is_empty()
    }
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.name, self.arity)?;
        if !self.inputs.is_empty() || !self.outputs.is_empty() {
            write!(f, " : ")?;
            if self.inputs.is_empty() {
                write!(f, "()")?;
            } else {
                write!(f, "(")?;
                for (i, inp) in self.inputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", inp)?;
                }
                write!(f, ")")?;
            }
            write!(f, " → ")?;
            if self.outputs.is_empty() {
                write!(f, "()")?;
            } else {
                write!(f, "(")?;
                for (i, out) in self.outputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", out)?;
                }
                write!(f, ")")?;
            }
        }
        Ok(())
    }
}

// ============================================================================
// WiringPlan
// ============================================================================

/// A wiring plan that enforces operadic constraints.
///
/// A `WiringPlan` connects inner operations to the input slots of an outer
/// operation. Validation ensures:
///
/// 1. Number of inner operations equals outer arity
/// 2. Each inner output shape matches the expected outer input shape
/// 3. No slot is wired twice
/// 4. All slots are wired
///
/// # Example
///
/// ```rust
/// use compositional_core::operad::{Operation, WiringPlan};
/// use compositional_core::Shape;
///
/// // Outer: expects 2 inputs
/// let outer = Operation::new("Combiner", 2)
///     .with_inputs(vec![Shape::f32_vector(10), Shape::f32_vector(10)])
///     .with_outputs(vec![Shape::f32_vector(20)]);
///
/// // Inner: two operations that produce outputs
/// let inner1 = Operation::new("Producer1", 0)
///     .with_outputs(vec![Shape::f32_vector(10)]);
/// let inner2 = Operation::new("Producer2", 0)
///     .with_outputs(vec![Shape::f32_vector(10)]);
///
/// let plan = WiringPlan::new(outer)
///     .with_inner(vec![inner1, inner2])
///     .with_wiring(vec![(0, 0), (1, 1)]);  // inner0 → slot0, inner1 → slot1
///
/// assert!(plan.validate().is_ok());
/// ```
#[derive(Debug, Clone)]
pub struct WiringPlan {
    /// The outer operation receiving inputs.
    pub outer: Operation,
    /// Inner operations providing inputs.
    pub inner: Vec<Operation>,
    /// Wiring: `(inner_op_idx, outer_slot_idx)`.
    pub wiring: Vec<(usize, usize)>,
}

impl WiringPlan {
    /// Create a new wiring plan with an outer operation.
    pub fn new(outer: Operation) -> Self {
        Self {
            outer,
            inner: Vec::new(),
            wiring: Vec::new(),
        }
    }

    /// Set inner operations.
    pub fn with_inner(mut self, inner: Vec<Operation>) -> Self {
        self.inner = inner;
        self
    }

    /// Add an inner operation.
    pub fn add_inner(mut self, op: Operation) -> Self {
        self.inner.push(op);
        self
    }

    /// Set wiring.
    pub fn with_wiring(mut self, wiring: Vec<(usize, usize)>) -> Self {
        self.wiring = wiring;
        self
    }

    /// Add a single wire.
    pub fn wire(mut self, inner_idx: usize, slot_idx: usize) -> Self {
        self.wiring.push((inner_idx, slot_idx));
        self
    }

    /// Auto-wire: wire inner operation i to slot i (for simple cases).
    pub fn auto_wire(mut self) -> Self {
        self.wiring = (0..self.inner.len()).map(|i| (i, i)).collect();
        self
    }

    /// Validate the wiring plan.
    ///
    /// Checks:
    /// 1. Number of inner operations equals outer arity
    /// 2. Each inner output shape matches expected outer input shape
    /// 3. No slot is wired twice
    /// 4. All slots are wired
    pub fn validate(&self) -> Result<(), OperadError> {
        // 1. Check arity
        if self.inner.len() != self.outer.arity {
            return Err(OperadError::ArityMismatch {
                operation: self.outer.name.clone(),
                expected: self.outer.arity,
                got: self.inner.len(),
            });
        }

        // 2. Check wiring validity
        let mut wired_slots: HashSet<usize> = HashSet::new();

        for &(inner_idx, slot_idx) in &self.wiring {
            // Check inner index is valid
            if inner_idx >= self.inner.len() {
                return Err(OperadError::InvalidWiringSource {
                    inner_idx,
                    max: self.inner.len(),
                });
            }

            // Check slot index is valid
            if slot_idx >= self.outer.arity {
                return Err(OperadError::InvalidWiringTarget {
                    slot_idx,
                    max: self.outer.arity,
                });
            }

            // 3. Check for duplicate wiring
            if !wired_slots.insert(slot_idx) {
                return Err(OperadError::DuplicateWiring { slot: slot_idx });
            }

            // 4. Check shape compatibility
            if let (Some(inner_output), Some(outer_input)) = (
                self.inner[inner_idx].output_shape(),
                self.outer.input_shape(slot_idx),
            ) {
                if !inner_output.is_compatible(outer_input) {
                    return Err(OperadError::ShapeMismatch {
                        slot: slot_idx,
                        expected: outer_input.clone(),
                        got: inner_output.clone(),
                    });
                }
            }
        }

        // 5. Check all slots are wired
        for slot in 0..self.outer.arity {
            if !wired_slots.contains(&slot) {
                return Err(OperadError::UnwiredSlot { slot });
            }
        }

        Ok(())
    }

    /// Convert to a Diagram (loses arity enforcement).
    ///
    /// The resulting diagram has the same structure but without the
    /// compile-time arity guarantees that operads provide.
    pub fn to_diagram(&self) -> Result<Diagram<OperadOp>, CoreError> {
        let mut diagram = Diagram::new();

        // Add outer operation as a node
        let outer_node = Node::new(
            OperadOp::Named(self.outer.name.clone()),
            self.outer
                .inputs
                .iter()
                .map(|s| Port::new(s.clone()))
                .collect(),
            self.outer
                .outputs
                .iter()
                .map(|s| Port::new(s.clone()))
                .collect(),
        );
        let outer_idx = diagram.add_node(outer_node);

        // Add inner operations as nodes
        let inner_indices: Vec<_> = self
            .inner
            .iter()
            .map(|op| {
                let node = Node::new(
                    OperadOp::Named(op.name.clone()),
                    op.inputs.iter().map(|s| Port::new(s.clone())).collect(),
                    op.outputs.iter().map(|s| Port::new(s.clone())).collect(),
                );
                diagram.add_node(node)
            })
            .collect();

        // Wire according to plan
        for &(inner_idx, slot_idx) in &self.wiring {
            let from_node = inner_indices[inner_idx];
            let to_node = outer_idx;
            // Connect output 0 of inner to input slot_idx of outer
            diagram.connect(from_node, 0, to_node, slot_idx)?;
        }

        // Set boundaries
        // Inputs: all inner operations' inputs (that aren't wired internally)
        let mut inputs = Vec::new();
        for (i, inner_op) in self.inner.iter().enumerate() {
            for port_idx in 0..inner_op.inputs.len() {
                inputs.push((inner_indices[i], port_idx));
            }
        }
        diagram.set_inputs(inputs);

        // Outputs: outer operation's outputs
        let outputs: Vec<_> = (0..self.outer.outputs.len())
            .map(|i| (outer_idx, i))
            .collect();
        diagram.set_outputs(outputs);

        Ok(diagram)
    }

    /// Get the output shapes of this wiring plan.
    pub fn output_shapes(&self) -> Vec<Shape> {
        self.outer.outputs.clone()
    }

    /// Get the input shapes (collected from all inner operations' inputs).
    pub fn input_shapes(&self) -> Vec<Shape> {
        self.inner
            .iter()
            .flat_map(|op| op.inputs.iter().cloned())
            .collect()
    }
}

impl fmt::Display for WiringPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "WiringPlan {{")?;
        writeln!(f, "  outer: {}", self.outer)?;
        writeln!(f, "  inner: [")?;
        for (i, op) in self.inner.iter().enumerate() {
            writeln!(f, "    [{}] {}", i, op)?;
        }
        writeln!(f, "  ]")?;
        writeln!(f, "  wiring: {:?}", self.wiring)?;
        write!(f, "}}")
    }
}

// ============================================================================
// OperadOp - Operation type for Diagram conversion
// ============================================================================

/// Operation type used when converting WiringPlan to Diagram.
#[derive(Debug, Clone, PartialEq)]
pub enum OperadOp {
    /// A named operation.
    Named(String),
}

// ============================================================================
// Operadic Composition
// ============================================================================

/// Compose two wiring plans: substitute the result of `inner` into a slot of `outer`.
///
/// This is the fundamental operadic composition operation:
/// Given `outer` expecting inputs at various slots, and `inner` producing outputs,
/// we can plug `inner`'s outputs into one of `outer`'s slots.
///
/// # Example
///
/// ```rust
/// use compositional_core::operad::{Operation, WiringPlan, compose_plans};
/// use compositional_core::Shape;
///
/// // Level 1: Simple producers
/// let p1 = Operation::new("Producer1", 0).with_outputs(vec![Shape::f32_vector(10)]);
/// let p2 = Operation::new("Producer2", 0).with_outputs(vec![Shape::f32_vector(10)]);
///
/// // Level 2: Combiner that takes 2 inputs
/// let combiner = Operation::new("Combiner", 2)
///     .with_inputs(vec![Shape::f32_vector(10), Shape::f32_vector(10)])
///     .with_outputs(vec![Shape::f32_vector(20)]);
///
/// let plan = WiringPlan::new(combiner)
///     .with_inner(vec![p1, p2])
///     .auto_wire();
///
/// assert!(plan.validate().is_ok());
/// ```
pub fn compose_plans(
    outer_plan: &WiringPlan,
    inner_plan: &WiringPlan,
    slot: usize,
) -> Result<WiringPlan, OperadError> {
    // Validate slot
    if slot >= outer_plan.outer.arity {
        return Err(OperadError::InvalidSlot { index: slot });
    }

    // The inner plan's outputs must match the outer's expected input at `slot`
    if let (Some(inner_out), Some(outer_in)) = (
        inner_plan.outer.output_shape(),
        outer_plan.outer.input_shape(slot),
    ) {
        if !inner_out.is_compatible(outer_in) {
            return Err(OperadError::ShapeMismatch {
                slot,
                expected: outer_in.clone(),
                got: inner_out.clone(),
            });
        }
    }

    // Create new outer with modified arity
    // The slot being filled is replaced by inner_plan's inputs
    let new_arity = outer_plan.outer.arity - 1 + inner_plan.outer.arity;

    let mut new_inputs = Vec::new();
    for (i, shape) in outer_plan.outer.inputs.iter().enumerate() {
        if i == slot {
            // Replace this slot with inner plan's inputs
            new_inputs.extend(inner_plan.outer.inputs.iter().cloned());
        } else {
            new_inputs.push(shape.clone());
        }
    }

    let new_outer = Operation {
        name: format!("{}∘{}", outer_plan.outer.name, inner_plan.outer.name),
        arity: new_arity,
        inputs: new_inputs,
        outputs: outer_plan.outer.outputs.clone(),
    };

    // Build new inner operations list
    // Operations before the slot stay the same (indices unchanged)
    // At the slot, we insert the inner_plan's inner operations
    // Operations after the slot have indices shifted

    let mut new_inner = Vec::new();
    let mut new_wiring = Vec::new();

    let mut current_slot = 0;
    for (i, op) in outer_plan.inner.iter().enumerate() {
        // Find which slot this inner op is wired to
        let wired_slot = outer_plan
            .wiring
            .iter()
            .find(|&&(inner_idx, _)| inner_idx == i)
            .map(|&(_, s)| s);

        if wired_slot == Some(slot) {
            // This is the slot we're replacing
            // Insert inner_plan's inners here
            let base_idx = new_inner.len();
            new_inner.extend(inner_plan.inner.iter().cloned());

            // Add inner_plan's wiring, adjusted for new indices
            for &(inner_idx, inner_slot) in &inner_plan.wiring {
                new_wiring.push((base_idx + inner_idx, current_slot + inner_slot));
            }
            current_slot += inner_plan.outer.arity;
        } else {
            // Keep this operation
            let new_idx = new_inner.len();
            new_inner.push(op.clone());
            if let Some(s) = wired_slot {
                let adjusted_slot = if s > slot {
                    s + inner_plan.outer.arity - 1
                } else {
                    s
                };
                new_wiring.push((new_idx, adjusted_slot));
            }
            current_slot += 1;
        }
    }

    Ok(WiringPlan {
        outer: new_outer,
        inner: new_inner,
        wiring: new_wiring,
    })
}

// ============================================================================
// Builder Helpers
// ============================================================================

/// Builder for creating operations with a fluent API.
pub struct OperationBuilder {
    name: String,
    arity: usize,
    inputs: Vec<Shape>,
    outputs: Vec<Shape>,
}

impl OperationBuilder {
    /// Start building an operation.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            arity: 0,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Set the arity.
    pub fn arity(mut self, arity: usize) -> Self {
        self.arity = arity;
        self
    }

    /// Add an input shape.
    pub fn input(mut self, shape: Shape) -> Self {
        self.inputs.push(shape);
        self.arity = self.inputs.len();
        self
    }

    /// Add an output shape.
    pub fn output(mut self, shape: Shape) -> Self {
        self.outputs.push(shape);
        self
    }

    /// Build the operation.
    pub fn build(self) -> Operation {
        Operation {
            name: self.name,
            arity: self.arity,
            inputs: self.inputs,
            outputs: self.outputs,
        }
    }
}

/// Convenience function to create an operation with the builder.
pub fn op(name: impl Into<String>) -> OperationBuilder {
    OperationBuilder::new(name)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn text_shape() -> Shape {
        Shape::f32_vector(256)
    }

    #[test]
    fn test_operation_creation() {
        let op = Operation::new("Test", 3);
        assert_eq!(op.name, "Test");
        assert_eq!(op.arity, 3);
        assert!(op.inputs.is_empty());
        assert!(op.outputs.is_empty());
    }

    #[test]
    fn test_operation_with_shapes() {
        let op = Operation::new("Agent", 2)
            .with_inputs(vec![text_shape(), text_shape()])
            .with_outputs(vec![text_shape()]);

        assert_eq!(op.arity, 2);
        assert_eq!(op.inputs.len(), 2);
        assert_eq!(op.outputs.len(), 1);
        assert!(op.is_fully_typed());
    }

    #[test]
    #[should_panic(expected = "inputs.len()")]
    fn test_operation_inputs_arity_mismatch() {
        Operation::new("Bad", 2).with_inputs(vec![text_shape()]); // Only 1 input for arity 2
    }

    #[test]
    fn test_wiring_plan_valid() {
        let outer = Operation::new("Agent", 3)
            .with_inputs(vec![text_shape(), text_shape(), text_shape()])
            .with_outputs(vec![text_shape()]);

        let inner1 = Operation::new("Search", 0).with_outputs(vec![text_shape()]);
        let inner2 = Operation::new("Calc", 0).with_outputs(vec![text_shape()]);
        let inner3 = Operation::new("Fetch", 0).with_outputs(vec![text_shape()]);

        let plan = WiringPlan::new(outer)
            .with_inner(vec![inner1, inner2, inner3])
            .with_wiring(vec![(0, 0), (1, 1), (2, 2)]);

        assert!(plan.validate().is_ok());
    }

    #[test]
    fn test_wiring_plan_auto_wire() {
        let outer = Operation::new("Agent", 2)
            .with_inputs(vec![text_shape(), text_shape()])
            .with_outputs(vec![text_shape()]);

        let inner1 = Operation::new("Tool1", 0).with_outputs(vec![text_shape()]);
        let inner2 = Operation::new("Tool2", 0).with_outputs(vec![text_shape()]);

        let plan = WiringPlan::new(outer)
            .with_inner(vec![inner1, inner2])
            .auto_wire();

        assert!(plan.validate().is_ok());
        assert_eq!(plan.wiring, vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn test_wiring_plan_arity_mismatch() {
        let outer = Operation::new("Agent", 3)
            .with_inputs(vec![text_shape(), text_shape(), text_shape()])
            .with_outputs(vec![text_shape()]);

        let inner1 = Operation::new("Search", 0).with_outputs(vec![text_shape()]);
        let inner2 = Operation::new("Calc", 0).with_outputs(vec![text_shape()]);
        // Missing third inner!

        let plan = WiringPlan::new(outer)
            .with_inner(vec![inner1, inner2])
            .with_wiring(vec![(0, 0), (1, 1)]);

        let err = plan.validate().unwrap_err();
        assert!(matches!(err, OperadError::ArityMismatch { .. }));
    }

    #[test]
    fn test_wiring_plan_shape_mismatch() {
        let outer = Operation::new("Agent", 2)
            .with_inputs(vec![Shape::f32_vector(100), Shape::f32_vector(200)])
            .with_outputs(vec![text_shape()]);

        let inner1 = Operation::new("Tool1", 0).with_outputs(vec![Shape::f32_vector(100)]);
        let inner2 = Operation::new("Tool2", 0).with_outputs(vec![Shape::f32_vector(100)]); // Wrong shape!

        let plan = WiringPlan::new(outer)
            .with_inner(vec![inner1, inner2])
            .auto_wire();

        let err = plan.validate().unwrap_err();
        assert!(matches!(err, OperadError::ShapeMismatch { slot: 1, .. }));
    }

    #[test]
    fn test_wiring_plan_duplicate_wiring() {
        let outer = Operation::new("Agent", 2)
            .with_inputs(vec![text_shape(), text_shape()])
            .with_outputs(vec![text_shape()]);

        let inner1 = Operation::new("Tool1", 0).with_outputs(vec![text_shape()]);
        let inner2 = Operation::new("Tool2", 0).with_outputs(vec![text_shape()]);

        let plan = WiringPlan::new(outer)
            .with_inner(vec![inner1, inner2])
            .with_wiring(vec![(0, 0), (1, 0)]); // Both wired to slot 0!

        let err = plan.validate().unwrap_err();
        assert!(matches!(err, OperadError::DuplicateWiring { slot: 0 }));
    }

    #[test]
    fn test_wiring_plan_unwired_slot() {
        let outer = Operation::new("Agent", 2)
            .with_inputs(vec![text_shape(), text_shape()])
            .with_outputs(vec![text_shape()]);

        let inner1 = Operation::new("Tool1", 0).with_outputs(vec![text_shape()]);
        let inner2 = Operation::new("Tool2", 0).with_outputs(vec![text_shape()]);

        let plan = WiringPlan::new(outer)
            .with_inner(vec![inner1, inner2])
            .with_wiring(vec![(0, 0)]); // Slot 1 not wired!

        let err = plan.validate().unwrap_err();
        assert!(matches!(err, OperadError::UnwiredSlot { slot: 1 }));
    }

    #[test]
    fn test_wiring_plan_to_diagram() {
        let outer = Operation::new("Combiner", 2)
            .with_inputs(vec![text_shape(), text_shape()])
            .with_outputs(vec![text_shape()]);

        let inner1 = Operation::new("Producer1", 0).with_outputs(vec![text_shape()]);
        let inner2 = Operation::new("Producer2", 0).with_outputs(vec![text_shape()]);

        let plan = WiringPlan::new(outer)
            .with_inner(vec![inner1, inner2])
            .auto_wire();

        let diagram = plan.to_diagram().unwrap();

        assert_eq!(diagram.node_count(), 3); // 2 inner + 1 outer
        assert_eq!(diagram.edge_count(), 2); // 2 wires
    }

    #[test]
    fn test_operation_builder() {
        let operation = op("MyAgent")
            .input(text_shape())
            .input(text_shape())
            .output(text_shape())
            .build();

        assert_eq!(operation.name, "MyAgent");
        assert_eq!(operation.arity, 2);
        assert_eq!(operation.inputs.len(), 2);
        assert_eq!(operation.outputs.len(), 1);
    }

    #[test]
    fn test_operation_display() {
        let op = Operation::new("Agent", 2)
            .with_inputs(vec![Shape::f32_vector(10), Shape::f32_vector(20)])
            .with_outputs(vec![Shape::f32_vector(30)]);

        let display = format!("{}", op);
        assert!(display.contains("Agent(2)"));
        assert!(display.contains("f32[10]"));
        assert!(display.contains("f32[20]"));
        assert!(display.contains("f32[30]"));
    }

    #[test]
    fn test_tool_use_pipeline() {
        // Real-world example: LLM tool-use pipeline
        let agent = Operation::new("ReasoningAgent", 3)
            .with_inputs(vec![
                Shape::f32_vector(512), // search result embedding
                Shape::f32_vector(512), // calculation result embedding
                Shape::f32_vector(512), // fetch result embedding
            ])
            .with_outputs(vec![Shape::f32_vector(1024)]); // response embedding

        let search = Operation::new("SearchTool", 0).with_outputs(vec![Shape::f32_vector(512)]);

        let calculator =
            Operation::new("CalculatorTool", 0).with_outputs(vec![Shape::f32_vector(512)]);

        let fetch = Operation::new("WebFetchTool", 0).with_outputs(vec![Shape::f32_vector(512)]);

        let plan = WiringPlan::new(agent)
            .with_inner(vec![search, calculator, fetch])
            .auto_wire();

        assert!(plan.validate().is_ok());
        println!("{}", plan);
    }

    #[test]
    fn test_multi_agent_orchestration() {
        // Orchestrator with 4 specialist agents
        let orchestrator = Operation::new("Orchestrator", 4)
            .with_inputs(vec![
                Shape::f32_vector(256),
                Shape::f32_vector(256),
                Shape::f32_vector(256),
                Shape::f32_vector(256),
            ])
            .with_outputs(vec![Shape::f32_vector(512)]);

        let researcher =
            Operation::new("ResearchAgent", 0).with_outputs(vec![Shape::f32_vector(256)]);

        let analyst = Operation::new("AnalystAgent", 0).with_outputs(vec![Shape::f32_vector(256)]);

        let writer = Operation::new("WriterAgent", 0).with_outputs(vec![Shape::f32_vector(256)]);

        let reviewer =
            Operation::new("ReviewerAgent", 0).with_outputs(vec![Shape::f32_vector(256)]);

        let plan = WiringPlan::new(orchestrator)
            .with_inner(vec![researcher, analyst, writer, reviewer])
            .auto_wire();

        assert!(plan.validate().is_ok());

        // Diagram conversion works
        let diagram = plan.to_diagram().unwrap();
        assert_eq!(diagram.node_count(), 5); // 4 agents + 1 orchestrator
    }
}
