//! # Diagrams - String Diagram Data Model
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
//! ## Composition (Sessions 4-5)
//!
//! - `then`: Sequential composition (categorical `;`)
//! - `tensor`: Parallel composition (monoidal `⊗`)
//!
//! The interchange law guarantees these compose correctly.

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
}
