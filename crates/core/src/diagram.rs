//! # Diagrams - String Diagram Data Model (Session 4)
//!
//! A diagram is a program: boxes (operations) connected by wires (data flow).
//! This is the core abstraction that all other crates build upon.
//!
//! ## Key Concepts
//!
//! - **Node**: An operation with typed input/output ports
//! - **Edge**: A wire connecting an output port to an input port
//! - **Diagram**: A directed graph of nodes and edges with boundary ports
//!
//! ## Composition
//!
//! - [`Diagram::then`]: Sequential composition (categorical `;`) — connect outputs to inputs
//! - [`Diagram::tensor`]: Parallel composition (monoidal `⊗`) — side by side
//!
//! ## Monoidal Category Laws
//!
//! The composition operations satisfy:
//! - **Associativity**: `(f ; g) ; h = f ; (g ; h)`
//! - **Identity**: `id ; f = f = f ; id`
//! - **Interchange**: `(f₁ ; g₁) ⊗ (f₂ ; g₂) = (f₁ ⊗ f₂) ; (g₁ ⊗ g₂)`
//!
//! ## Example
//!
//! ```rust
//! use compositional_core::diagram::{Diagram, Node, Port};
//! use compositional_core::Shape;
//!
//! #[derive(Debug, Clone)]
//! enum Op { F, G }
//!
//! // Create f: A → B
//! let f = Diagram::from_node(Node::new(
//!     Op::F,
//!     vec![Port::new(Shape::f32_vector(10))],  // input A
//!     vec![Port::new(Shape::f32_vector(20))],  // output B
//! ));
//!
//! // Create g: B → C
//! let g = Diagram::from_node(Node::new(
//!     Op::G,
//!     vec![Port::new(Shape::f32_vector(20))],  // input B
//!     vec![Port::new(Shape::f32_vector(30))],  // output C
//! ));
//!
//! // Sequential composition: f ; g : A → C
//! let fg = f.then(g).expect("shapes should match");
//! assert_eq!(fg.node_count(), 2);
//! ```

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use std::fmt;

use crate::error::CoreError;
use crate::shape::Shape;

/// A port is a typed connection point on a node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Port {
    pub shape: Shape,
}

impl Port {
    pub fn new(shape: Shape) -> Self {
        Self { shape }
    }
}

/// A node in the diagram, parameterized by operation type.
///
/// The generic `O` allows different domains to define their own operations:
/// - `DiffOp` for autodiff (Add, MatMul, ReLU, etc.)
/// - `ProbOp` for probability (Sample, Condition, etc.)
/// - `GrammarOp` for NLP, etc.
#[derive(Debug, Clone)]
pub struct Node<O> {
    /// The operation this node performs
    pub op: O,
    /// Input ports (data flows in)
    pub inputs: Vec<Port>,
    /// Output ports (data flows out)
    pub outputs: Vec<Port>,
}

impl<O> Node<O> {
    pub fn new(op: O, inputs: Vec<Port>, outputs: Vec<Port>) -> Self {
        Self {
            op,
            inputs,
            outputs,
        }
    }

    /// Number of input ports
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// Number of output ports
    pub fn num_outputs(&self) -> usize {
        self.outputs.len()
    }
}

/// An edge connects an output port of one node to an input port of another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edge {
    /// Index of the output port on the source node
    pub from_port: usize,
    /// Index of the input port on the target node
    pub to_port: usize,
}

impl Edge {
    pub fn new(from_port: usize, to_port: usize) -> Self {
        Self { from_port, to_port }
    }
}

/// A boundary port reference: (node index, port index)
pub type BoundaryPort = (NodeIndex, usize);

/// A diagram is a directed graph of nodes connected by edges,
/// with explicit input and output boundaries.
///
/// The boundaries define how this diagram composes with others:
/// - `inputs`: External data enters through these ports
/// - `outputs`: Results exit through these ports
#[derive(Debug, Clone)]
pub struct Diagram<O> {
    /// The underlying graph structure
    pub graph: DiGraph<Node<O>, Edge>,
    /// Boundary input ports (where external data enters)
    pub inputs: Vec<BoundaryPort>,
    /// Boundary output ports (where results exit)
    pub outputs: Vec<BoundaryPort>,
}

impl<O: Clone> Diagram<O> {
    /// Create a new empty diagram.
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Add a node to the diagram, returning its index.
    pub fn add_node(&mut self, node: Node<O>) -> NodeIndex {
        self.graph.add_node(node)
    }

    /// Connect an output port of one node to an input port of another.
    ///
    /// Returns an error if:
    /// - Port indices are out of bounds
    /// - Shapes don't match
    pub fn connect(
        &mut self,
        from_node: NodeIndex,
        from_port: usize,
        to_node: NodeIndex,
        to_port: usize,
    ) -> Result<(), CoreError> {
        // Validate port indices
        let from_node_data =
            self.graph
                .node_weight(from_node)
                .ok_or_else(|| CoreError::ValidationError {
                    reason: "Source node not found".to_string(),
                })?;

        if from_port >= from_node_data.outputs.len() {
            return Err(CoreError::InvalidPort {
                index: from_port,
                count: from_node_data.outputs.len(),
            });
        }

        let from_shape = from_node_data.outputs[from_port].shape.clone();

        let to_node_data =
            self.graph
                .node_weight(to_node)
                .ok_or_else(|| CoreError::ValidationError {
                    reason: "Target node not found".to_string(),
                })?;

        if to_port >= to_node_data.inputs.len() {
            return Err(CoreError::InvalidPort {
                index: to_port,
                count: to_node_data.inputs.len(),
            });
        }

        let to_shape = &to_node_data.inputs[to_port].shape;

        // Check shape compatibility
        if !from_shape.is_compatible(to_shape) {
            return Err(CoreError::ShapeMismatch {
                expected: to_shape.clone(),
                got: from_shape,
            });
        }

        // Add the edge
        self.graph
            .add_edge(from_node, to_node, Edge::new(from_port, to_port));
        Ok(())
    }

    /// Set the input boundary ports.
    pub fn set_inputs(&mut self, inputs: Vec<BoundaryPort>) {
        self.inputs = inputs;
    }

    /// Set the output boundary ports.
    pub fn set_outputs(&mut self, outputs: Vec<BoundaryPort>) {
        self.outputs = outputs;
    }

    /// Get the shapes of all input boundary ports.
    pub fn input_shapes(&self) -> Vec<Shape> {
        self.inputs
            .iter()
            .filter_map(|(node_idx, port_idx)| {
                self.graph
                    .node_weight(*node_idx)
                    .and_then(|n| n.inputs.get(*port_idx))
                    .map(|p| p.shape.clone())
            })
            .collect()
    }

    /// Get the shapes of all output boundary ports.
    pub fn output_shapes(&self) -> Vec<Shape> {
        self.outputs
            .iter()
            .filter_map(|(node_idx, port_idx)| {
                self.graph
                    .node_weight(*node_idx)
                    .and_then(|n| n.outputs.get(*port_idx))
                    .map(|p| p.shape.clone())
            })
            .collect()
    }

    /// Validate the diagram's internal consistency.
    ///
    /// Checks:
    /// - All edges connect compatible shapes
    /// - Boundary ports exist
    pub fn validate(&self) -> Result<(), CoreError> {
        // Check all edges have compatible shapes
        for edge_ref in self.graph.edge_references() {
            let from_node = self.graph.node_weight(edge_ref.source()).ok_or_else(|| {
                CoreError::ValidationError {
                    reason: "Edge source node not found".to_string(),
                }
            })?;
            let to_node = self.graph.node_weight(edge_ref.target()).ok_or_else(|| {
                CoreError::ValidationError {
                    reason: "Edge target node not found".to_string(),
                }
            })?;

            let edge = edge_ref.weight();

            let from_shape =
                from_node
                    .outputs
                    .get(edge.from_port)
                    .ok_or(CoreError::InvalidPort {
                        index: edge.from_port,
                        count: from_node.outputs.len(),
                    })?;

            let to_shape = to_node
                .inputs
                .get(edge.to_port)
                .ok_or(CoreError::InvalidPort {
                    index: edge.to_port,
                    count: to_node.inputs.len(),
                })?;

            if !from_shape.shape.is_compatible(&to_shape.shape) {
                return Err(CoreError::ShapeMismatch {
                    expected: to_shape.shape.clone(),
                    got: from_shape.shape.clone(),
                });
            }
        }

        // Check boundary ports exist
        for (node_idx, port_idx) in &self.inputs {
            let node =
                self.graph
                    .node_weight(*node_idx)
                    .ok_or_else(|| CoreError::ValidationError {
                        reason: format!("Input boundary node {:?} not found", node_idx),
                    })?;
            if *port_idx >= node.inputs.len() {
                return Err(CoreError::InvalidPort {
                    index: *port_idx,
                    count: node.inputs.len(),
                });
            }
        }

        for (node_idx, port_idx) in &self.outputs {
            let node =
                self.graph
                    .node_weight(*node_idx)
                    .ok_or_else(|| CoreError::ValidationError {
                        reason: format!("Output boundary node {:?} not found", node_idx),
                    })?;
            if *port_idx >= node.outputs.len() {
                return Err(CoreError::InvalidPort {
                    index: *port_idx,
                    count: node.outputs.len(),
                });
            }
        }

        Ok(())
    }

    /// Number of nodes in the diagram.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of edges in the diagram.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    // ========================================================================
    // Diagram Constructors
    // ========================================================================

    /// Create a diagram from a single node.
    ///
    /// The node's input ports become the diagram's input boundary,
    /// and output ports become the output boundary.
    pub fn from_node(node: Node<O>) -> Self {
        let mut diagram = Self::new();
        let inputs: Vec<_> = (0..node.inputs.len()).collect();
        let outputs: Vec<_> = (0..node.outputs.len()).collect();
        let idx = diagram.add_node(node);

        diagram.inputs = inputs.into_iter().map(|i| (idx, i)).collect();
        diagram.outputs = outputs.into_iter().map(|i| (idx, i)).collect();

        diagram
    }

    /// Create an identity diagram for the given shapes.
    ///
    /// An identity diagram passes inputs directly to outputs without modification.
    /// This requires a "passthrough" operation that your Op type must support.
    ///
    /// For a true identity, use `identity_wire` which creates no nodes.
    pub fn identity_wires(_shapes: Vec<Shape>) -> Self {
        let mut diagram = Self::new();
        // Identity has no nodes - just matching inputs/outputs
        // We represent this as an empty diagram with shape info stored in boundaries
        // The actual "wiring" happens during composition
        diagram.inputs = Vec::new();
        diagram.outputs = Vec::new();
        // Store shapes for compatibility checking
        diagram
    }

    // ========================================================================
    // Sequential Composition: f ; g (then)
    // ========================================================================

    /// Sequential composition: `self ; other`
    ///
    /// Connects the outputs of `self` to the inputs of `other`.
    ///
    /// ```text
    ///    self: A → B       other: B → C
    ///
    ///    ┌──────┐          ┌───────┐
    /// ───│ self │───  ;  ───│ other │───
    ///    └──────┘          └───────┘
    ///
    ///    Result: A → C
    ///
    ///    ┌──────┐    ┌───────┐
    /// ───│ self │────│ other │───
    ///    └──────┘    └───────┘
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `ShapeMismatch` if the output shapes of `self` don't match
    /// the input shapes of `other`.
    pub fn then(self, other: Self) -> Result<Self, CoreError> {
        let self_outputs = self.output_shapes();
        let other_inputs = other.input_shapes();

        // Check that output count matches input count
        if self_outputs.len() != other_inputs.len() {
            return Err(CoreError::ValidationError {
                reason: format!(
                    "Cannot compose: self has {} outputs, other has {} inputs",
                    self_outputs.len(),
                    other_inputs.len()
                ),
            });
        }

        // Check shape compatibility
        for (out_shape, in_shape) in self_outputs.iter().zip(other_inputs.iter()) {
            if !out_shape.is_compatible(in_shape) {
                return Err(CoreError::ShapeMismatch {
                    expected: in_shape.clone(),
                    got: out_shape.clone(),
                });
            }
        }

        // Build the composed diagram
        let mut result = Diagram::new();

        // Maps from old node indices to new node indices
        let mut self_node_map: std::collections::HashMap<NodeIndex, NodeIndex> =
            std::collections::HashMap::new();
        let mut other_node_map: std::collections::HashMap<NodeIndex, NodeIndex> =
            std::collections::HashMap::new();

        // Copy nodes from self
        for node_idx in self.graph.node_indices() {
            let node = self.graph.node_weight(node_idx).unwrap().clone();
            let new_idx = result.add_node(node);
            self_node_map.insert(node_idx, new_idx);
        }

        // Copy nodes from other
        for node_idx in other.graph.node_indices() {
            let node = other.graph.node_weight(node_idx).unwrap().clone();
            let new_idx = result.add_node(node);
            other_node_map.insert(node_idx, new_idx);
        }

        // Copy edges from self
        for edge_ref in self.graph.edge_references() {
            let from = self_node_map[&edge_ref.source()];
            let to = self_node_map[&edge_ref.target()];
            let edge = edge_ref.weight().clone();
            result.graph.add_edge(from, to, edge);
        }

        // Copy edges from other
        for edge_ref in other.graph.edge_references() {
            let from = other_node_map[&edge_ref.source()];
            let to = other_node_map[&edge_ref.target()];
            let edge = edge_ref.weight().clone();
            result.graph.add_edge(from, to, edge);
        }

        // Connect self's outputs to other's inputs
        for (self_out, other_in) in self.outputs.iter().zip(other.inputs.iter()) {
            let from_node = self_node_map[&self_out.0];
            let from_port = self_out.1;
            let to_node = other_node_map[&other_in.0];
            let to_port = other_in.1;

            result
                .graph
                .add_edge(from_node, to_node, Edge::new(from_port, to_port));
        }

        // Set boundaries: inputs from self, outputs from other
        result.inputs = self
            .inputs
            .iter()
            .map(|(idx, port)| (self_node_map[idx], *port))
            .collect();

        result.outputs = other
            .outputs
            .iter()
            .map(|(idx, port)| (other_node_map[idx], *port))
            .collect();

        Ok(result)
    }

    // ========================================================================
    // Parallel Composition: f ⊗ g (tensor)
    // ========================================================================

    /// Parallel composition: `self ⊗ other`
    ///
    /// Places `self` and `other` side by side without connecting them.
    ///
    /// ```text
    ///    self: A → B       other: C → D
    ///
    ///    ┌──────┐
    /// ───│ self │───
    ///    └──────┘          ⊗
    ///    ┌───────┐
    /// ───│ other │───
    ///    └───────┘
    ///
    ///    Result: (A, C) → (B, D)
    ///
    ///    ┌──────┐
    /// ───│ self │───
    ///    └──────┘
    ///    ┌───────┐
    /// ───│ other │───
    ///    └───────┘
    /// ```
    ///
    /// The resulting diagram has:
    /// - Inputs: self.inputs ++ other.inputs
    /// - Outputs: self.outputs ++ other.outputs
    pub fn tensor(self, other: Self) -> Self {
        let mut result = Diagram::new();

        // Maps from old node indices to new node indices
        let mut self_node_map: std::collections::HashMap<NodeIndex, NodeIndex> =
            std::collections::HashMap::new();
        let mut other_node_map: std::collections::HashMap<NodeIndex, NodeIndex> =
            std::collections::HashMap::new();

        // Copy nodes from self
        for node_idx in self.graph.node_indices() {
            let node = self.graph.node_weight(node_idx).unwrap().clone();
            let new_idx = result.add_node(node);
            self_node_map.insert(node_idx, new_idx);
        }

        // Copy nodes from other
        for node_idx in other.graph.node_indices() {
            let node = other.graph.node_weight(node_idx).unwrap().clone();
            let new_idx = result.add_node(node);
            other_node_map.insert(node_idx, new_idx);
        }

        // Copy edges from self
        for edge_ref in self.graph.edge_references() {
            let from = self_node_map[&edge_ref.source()];
            let to = self_node_map[&edge_ref.target()];
            let edge = edge_ref.weight().clone();
            result.graph.add_edge(from, to, edge);
        }

        // Copy edges from other
        for edge_ref in other.graph.edge_references() {
            let from = other_node_map[&edge_ref.source()];
            let to = other_node_map[&edge_ref.target()];
            let edge = edge_ref.weight().clone();
            result.graph.add_edge(from, to, edge);
        }

        // Concatenate boundaries
        let mut inputs: Vec<BoundaryPort> = self
            .inputs
            .iter()
            .map(|(idx, port)| (self_node_map[idx], *port))
            .collect();
        inputs.extend(
            other
                .inputs
                .iter()
                .map(|(idx, port)| (other_node_map[idx], *port)),
        );

        let mut outputs: Vec<BoundaryPort> = self
            .outputs
            .iter()
            .map(|(idx, port)| (self_node_map[idx], *port))
            .collect();
        outputs.extend(
            other
                .outputs
                .iter()
                .map(|(idx, port)| (other_node_map[idx], *port)),
        );

        result.inputs = inputs;
        result.outputs = outputs;

        result
    }

    // ========================================================================
    // Utility Methods
    // ========================================================================

    /// Check if this diagram can be sequentially composed with another.
    ///
    /// Returns true if `self.output_shapes() == other.input_shapes()`.
    pub fn can_compose_with(&self, other: &Self) -> bool {
        let self_outputs = self.output_shapes();
        let other_inputs = other.input_shapes();

        if self_outputs.len() != other_inputs.len() {
            return false;
        }

        self_outputs
            .iter()
            .zip(other_inputs.iter())
            .all(|(a, b)| a.is_compatible(b))
    }

    /// Returns true if this is an "empty" diagram (no nodes).
    pub fn is_empty(&self) -> bool {
        self.graph.node_count() == 0
    }
}

impl<O: Clone> Default for Diagram<O> {
    fn default() -> Self {
        Self::new()
    }
}

impl<O: Clone + fmt::Debug> fmt::Display for Diagram<O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Diagram({} nodes, {} edges)",
            self.node_count(),
            self.edge_count()
        )?;
        writeln!(f, "  Inputs: {:?}", self.inputs)?;
        writeln!(f, "  Outputs: {:?}", self.outputs)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A simple test operation
    #[derive(Debug, Clone)]
    enum TestOp {
        Add,
        Mul,
    }

    #[test]
    fn test_create_diagram() {
        let diagram: Diagram<TestOp> = Diagram::new();
        assert_eq!(diagram.node_count(), 0);
        assert_eq!(diagram.edge_count(), 0);
    }

    #[test]
    fn test_add_node() {
        let mut diagram: Diagram<TestOp> = Diagram::new();

        let node = Node::new(
            TestOp::Add,
            vec![
                Port::new(Shape::f32_scalar()),
                Port::new(Shape::f32_scalar()),
            ],
            vec![Port::new(Shape::f32_scalar())],
        );

        let idx = diagram.add_node(node);
        assert_eq!(diagram.node_count(), 1);
        assert!(diagram.graph.node_weight(idx).is_some());
    }

    #[test]
    fn test_connect_matching_shapes() {
        let mut diagram: Diagram<TestOp> = Diagram::new();

        let node1 = Node::new(
            TestOp::Add,
            vec![
                Port::new(Shape::f32_scalar()),
                Port::new(Shape::f32_scalar()),
            ],
            vec![Port::new(Shape::f32_scalar())],
        );

        let node2 = Node::new(
            TestOp::Mul,
            vec![
                Port::new(Shape::f32_scalar()),
                Port::new(Shape::f32_scalar()),
            ],
            vec![Port::new(Shape::f32_scalar())],
        );

        let idx1 = diagram.add_node(node1);
        let idx2 = diagram.add_node(node2);

        // Connect output 0 of node1 to input 0 of node2
        let result = diagram.connect(idx1, 0, idx2, 0);
        assert!(result.is_ok());
        assert_eq!(diagram.edge_count(), 1);
    }

    #[test]
    fn test_connect_mismatched_shapes() {
        let mut diagram: Diagram<TestOp> = Diagram::new();

        let node1 = Node::new(
            TestOp::Add,
            vec![],
            vec![Port::new(Shape::f32_scalar())], // outputs scalar
        );

        let node2 = Node::new(
            TestOp::Mul,
            vec![Port::new(Shape::f32_vector(10))], // expects vector
            vec![],
        );

        let idx1 = diagram.add_node(node1);
        let idx2 = diagram.add_node(node2);

        let result = diagram.connect(idx1, 0, idx2, 0);
        assert!(matches!(result, Err(CoreError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_validate_valid_diagram() {
        let mut diagram: Diagram<TestOp> = Diagram::new();

        let node = Node::new(
            TestOp::Add,
            vec![Port::new(Shape::f32_scalar())],
            vec![Port::new(Shape::f32_scalar())],
        );

        let idx = diagram.add_node(node);
        diagram.set_inputs(vec![(idx, 0)]);
        diagram.set_outputs(vec![(idx, 0)]);

        assert!(diagram.validate().is_ok());
    }

    // ========================================================================
    // Session 4: Composition Tests
    // ========================================================================

    #[test]
    fn test_from_node() {
        let node = Node::new(
            TestOp::Add,
            vec![
                Port::new(Shape::f32_scalar()),
                Port::new(Shape::f32_scalar()),
            ],
            vec![Port::new(Shape::f32_scalar())],
        );

        let diagram = Diagram::from_node(node);

        assert_eq!(diagram.node_count(), 1);
        assert_eq!(diagram.inputs.len(), 2); // 2 input ports
        assert_eq!(diagram.outputs.len(), 1); // 1 output port
        assert_eq!(diagram.input_shapes().len(), 2);
        assert_eq!(diagram.output_shapes().len(), 1);
    }

    #[test]
    fn test_sequential_composition_basic() {
        // f: scalar → vector(10)
        let f = Diagram::from_node(Node::new(
            TestOp::Add,
            vec![Port::new(Shape::f32_scalar())],
            vec![Port::new(Shape::f32_vector(10))],
        ));

        // g: vector(10) → vector(20)
        let g = Diagram::from_node(Node::new(
            TestOp::Mul,
            vec![Port::new(Shape::f32_vector(10))],
            vec![Port::new(Shape::f32_vector(20))],
        ));

        // f ; g : scalar → vector(20)
        let fg = f.then(g).expect("composition should succeed");

        assert_eq!(fg.node_count(), 2);
        assert_eq!(fg.edge_count(), 1); // one connection between f and g
        assert_eq!(fg.input_shapes(), vec![Shape::f32_scalar()]);
        assert_eq!(fg.output_shapes(), vec![Shape::f32_vector(20)]);
    }

    #[test]
    fn test_sequential_composition_shape_mismatch() {
        // f: scalar → vector(10)
        let f = Diagram::from_node(Node::new(
            TestOp::Add,
            vec![Port::new(Shape::f32_scalar())],
            vec![Port::new(Shape::f32_vector(10))],
        ));

        // g: vector(20) → scalar (WRONG INPUT SIZE)
        let g = Diagram::from_node(Node::new(
            TestOp::Mul,
            vec![Port::new(Shape::f32_vector(20))],
            vec![Port::new(Shape::f32_scalar())],
        ));

        let result = f.then(g);
        assert!(matches!(result, Err(CoreError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_sequential_composition_arity_mismatch() {
        // f: scalar → (vector(10), vector(10)) -- 2 outputs
        let f = Diagram::from_node(Node::new(
            TestOp::Add,
            vec![Port::new(Shape::f32_scalar())],
            vec![
                Port::new(Shape::f32_vector(10)),
                Port::new(Shape::f32_vector(10)),
            ],
        ));

        // g: vector(10) → scalar -- 1 input (doesn't match 2 outputs)
        let g = Diagram::from_node(Node::new(
            TestOp::Mul,
            vec![Port::new(Shape::f32_vector(10))],
            vec![Port::new(Shape::f32_scalar())],
        ));

        let result = f.then(g);
        assert!(matches!(result, Err(CoreError::ValidationError { .. })));
    }

    #[test]
    fn test_sequential_composition_chain() {
        // f: A → B
        let f = Diagram::from_node(Node::new(
            TestOp::Add,
            vec![Port::new(Shape::f32_vector(10))],
            vec![Port::new(Shape::f32_vector(20))],
        ));

        // g: B → C
        let g = Diagram::from_node(Node::new(
            TestOp::Mul,
            vec![Port::new(Shape::f32_vector(20))],
            vec![Port::new(Shape::f32_vector(30))],
        ));

        // h: C → D
        let h = Diagram::from_node(Node::new(
            TestOp::Add,
            vec![Port::new(Shape::f32_vector(30))],
            vec![Port::new(Shape::f32_vector(40))],
        ));

        // (f ; g) ; h
        let fg = f.clone().then(g.clone()).unwrap();
        let fgh = fg.then(h.clone()).unwrap();

        assert_eq!(fgh.node_count(), 3);
        assert_eq!(fgh.input_shapes(), vec![Shape::f32_vector(10)]);
        assert_eq!(fgh.output_shapes(), vec![Shape::f32_vector(40)]);

        // f ; (g ; h) -- should give same result (associativity)
        let gh = g.then(h).unwrap();
        let f_gh = f.then(gh).unwrap();

        assert_eq!(f_gh.node_count(), 3);
        assert_eq!(f_gh.input_shapes(), vec![Shape::f32_vector(10)]);
        assert_eq!(f_gh.output_shapes(), vec![Shape::f32_vector(40)]);
    }

    #[test]
    fn test_parallel_composition_basic() {
        // f: A → B
        let f = Diagram::from_node(Node::new(
            TestOp::Add,
            vec![Port::new(Shape::f32_vector(10))],
            vec![Port::new(Shape::f32_vector(20))],
        ));

        // g: C → D
        let g = Diagram::from_node(Node::new(
            TestOp::Mul,
            vec![Port::new(Shape::f32_vector(30))],
            vec![Port::new(Shape::f32_vector(40))],
        ));

        // f ⊗ g : (A, C) → (B, D)
        let fg = f.tensor(g);

        assert_eq!(fg.node_count(), 2);
        assert_eq!(fg.edge_count(), 0); // no connections between parallel nodes
        assert_eq!(
            fg.input_shapes(),
            vec![Shape::f32_vector(10), Shape::f32_vector(30)]
        );
        assert_eq!(
            fg.output_shapes(),
            vec![Shape::f32_vector(20), Shape::f32_vector(40)]
        );
    }

    #[test]
    fn test_parallel_composition_preserves_internal_edges() {
        // Create a diagram with internal edges
        let mut f: Diagram<TestOp> = Diagram::new();
        let n1 = f.add_node(Node::new(
            TestOp::Add,
            vec![Port::new(Shape::f32_scalar())],
            vec![Port::new(Shape::f32_vector(10))],
        ));
        let n2 = f.add_node(Node::new(
            TestOp::Mul,
            vec![Port::new(Shape::f32_vector(10))],
            vec![Port::new(Shape::f32_vector(20))],
        ));
        f.connect(n1, 0, n2, 0).unwrap();
        f.set_inputs(vec![(n1, 0)]);
        f.set_outputs(vec![(n2, 0)]);

        // Simple single-node diagram
        let g = Diagram::from_node(Node::new(
            TestOp::Add,
            vec![Port::new(Shape::f32_scalar())],
            vec![Port::new(Shape::f32_scalar())],
        ));

        let fg = f.tensor(g);

        assert_eq!(fg.node_count(), 3);
        assert_eq!(fg.edge_count(), 1); // internal edge from f is preserved
    }

    #[test]
    fn test_can_compose_with() {
        let f = Diagram::from_node(Node::new(
            TestOp::Add,
            vec![Port::new(Shape::f32_scalar())],
            vec![Port::new(Shape::f32_vector(10))],
        ));

        let g_compatible = Diagram::from_node(Node::new(
            TestOp::Mul,
            vec![Port::new(Shape::f32_vector(10))],
            vec![Port::new(Shape::f32_scalar())],
        ));

        let g_incompatible = Diagram::from_node(Node::new(
            TestOp::Mul,
            vec![Port::new(Shape::f32_vector(20))], // wrong size
            vec![Port::new(Shape::f32_scalar())],
        ));

        assert!(f.can_compose_with(&g_compatible));
        assert!(!f.can_compose_with(&g_incompatible));
    }

    #[test]
    fn test_is_empty() {
        let empty: Diagram<TestOp> = Diagram::new();
        assert!(empty.is_empty());

        let non_empty = Diagram::from_node(Node::new(
            TestOp::Add,
            vec![Port::new(Shape::f32_scalar())],
            vec![Port::new(Shape::f32_scalar())],
        ));
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_multi_port_sequential_composition() {
        // f: (A, B) → (C, D) -- 2 inputs, 2 outputs
        let f = Diagram::from_node(Node::new(
            TestOp::Add,
            vec![
                Port::new(Shape::f32_vector(10)),
                Port::new(Shape::f32_vector(20)),
            ],
            vec![
                Port::new(Shape::f32_vector(30)),
                Port::new(Shape::f32_vector(40)),
            ],
        ));

        // g: (C, D) → (E, F) -- 2 inputs, 2 outputs
        let g = Diagram::from_node(Node::new(
            TestOp::Mul,
            vec![
                Port::new(Shape::f32_vector(30)),
                Port::new(Shape::f32_vector(40)),
            ],
            vec![
                Port::new(Shape::f32_vector(50)),
                Port::new(Shape::f32_vector(60)),
            ],
        ));

        let fg = f.then(g).expect("composition should succeed");

        assert_eq!(fg.node_count(), 2);
        assert_eq!(fg.edge_count(), 2); // two connections (one per output/input pair)
        assert_eq!(
            fg.input_shapes(),
            vec![Shape::f32_vector(10), Shape::f32_vector(20)]
        );
        assert_eq!(
            fg.output_shapes(),
            vec![Shape::f32_vector(50), Shape::f32_vector(60)]
        );
    }

    #[test]
    fn test_interchange_law() {
        // Interchange law: (f₁ ; g₁) ⊗ (f₂ ; g₂) = (f₁ ⊗ f₂) ; (g₁ ⊗ g₂)
        //
        // f₁: A → B
        let f1 = Diagram::from_node(Node::new(
            TestOp::Add,
            vec![Port::new(Shape::f32_vector(10))],
            vec![Port::new(Shape::f32_vector(20))],
        ));

        // g₁: B → C
        let g1 = Diagram::from_node(Node::new(
            TestOp::Mul,
            vec![Port::new(Shape::f32_vector(20))],
            vec![Port::new(Shape::f32_vector(30))],
        ));

        // f₂: D → E
        let f2 = Diagram::from_node(Node::new(
            TestOp::Add,
            vec![Port::new(Shape::f32_vector(40))],
            vec![Port::new(Shape::f32_vector(50))],
        ));

        // g₂: E → F
        let g2 = Diagram::from_node(Node::new(
            TestOp::Mul,
            vec![Port::new(Shape::f32_vector(50))],
            vec![Port::new(Shape::f32_vector(60))],
        ));

        // Left side: (f₁ ; g₁) ⊗ (f₂ ; g₂)
        let f1g1 = f1.clone().then(g1.clone()).unwrap();
        let f2g2 = f2.clone().then(g2.clone()).unwrap();
        let left = f1g1.tensor(f2g2);

        // Right side: (f₁ ⊗ f₂) ; (g₁ ⊗ g₂)
        let f1_f2 = f1.tensor(f2);
        let g1_g2 = g1.tensor(g2);
        let right = f1_f2.then(g1_g2).unwrap();

        // Both should have same boundaries (interchange law)
        assert_eq!(left.input_shapes(), right.input_shapes());
        assert_eq!(left.output_shapes(), right.output_shapes());
        assert_eq!(left.node_count(), right.node_count()); // 4 nodes each
    }
}
