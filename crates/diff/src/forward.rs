//! # Forward Evaluation (Session 8)
//!
//! This module implements forward evaluation of computation graphs—the
//! deterministic semantics that maps `(diagram, inputs) → outputs`.
//!
//! ## Key Concepts
//!
//! - **Topological order**: Process nodes so dependencies are computed first
//! - **Value cache**: Store intermediate results for each node
//! - **Edge routing**: Track which output port connects to which input port
//!
//! ## Example
//!
//! ```rust
//! use compositional_diff::forward::DiffGraph;
//! use compositional_diff::ops::RTensor;
//!
//! // Build: y = ReLU(a + b)
//! let mut graph = DiffGraph::new();
//! let a = graph.input(0, vec![3]);           // input vector of size 3
//! let b = graph.input(1, vec![3]);           // another input vector
//! let sum = graph.add(a, b);                 // a + b
//! let y = graph.relu(sum);                   // ReLU(a + b)
//! graph.mark_output(y);
//!
//! // Evaluate
//! let input_a = RTensor::vector(vec![-1.0, 2.0, -3.0]);
//! let input_b = RTensor::vector(vec![2.0, -1.0, 4.0]);
//! let outputs = graph.forward(&[input_a, input_b]);
//!
//! // Result: ReLU([-1+2, 2-1, -3+4]) = ReLU([1, 1, 1]) = [1, 1, 1]
//! assert_eq!(outputs[0].data, vec![1.0, 1.0, 1.0]);
//! ```

use crate::ops::{DiffOp, RTensor};
use compositional_core::{Diagram, Node, Port, Shape};
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use std::collections::{HashMap, VecDeque};

/// A computation graph for differentiable operations.
///
/// Wraps `Diagram<DiffOp>` with convenient builder methods and
/// forward evaluation.
pub struct DiffGraph {
    /// The underlying diagram
    pub diagram: Diagram<DiffOp>,
    /// Maps input indices to their node indices
    pub(crate) input_nodes: HashMap<usize, NodeIndex>,
    /// Output node indices (in order)
    pub(crate) output_nodes: Vec<NodeIndex>,
}

impl DiffGraph {
    /// Create a new empty computation graph.
    pub fn new() -> Self {
        Self {
            diagram: Diagram::new(),
            input_nodes: HashMap::new(),
            output_nodes: Vec::new(),
        }
    }

    /// Add an input node to the graph.
    ///
    /// The `index` identifies which boundary input this corresponds to.
    /// The `shape` specifies the expected tensor shape.
    pub fn input(&mut self, index: usize, shape: Vec<usize>) -> NodeIndex {
        let core_shape = vec_to_shape(&shape);
        let node = Node::new(
            DiffOp::Input { index },
            vec![Port::new(core_shape.clone())],
            vec![Port::new(core_shape)],
        );
        let idx = self.diagram.add_node(node);
        self.input_nodes.insert(index, idx);

        // Also add to diagram's boundary inputs
        self.diagram.inputs.push((idx, 0));

        idx
    }

    /// Add a constant node.
    pub fn constant(&mut self, value: f32) -> NodeIndex {
        let node = Node::new(
            DiffOp::Const { value },
            vec![],
            vec![Port::new(Shape::f32_scalar())],
        );
        self.diagram.add_node(node)
    }

    /// Add an addition node: a + b
    pub fn add(&mut self, a: NodeIndex, b: NodeIndex) -> NodeIndex {
        let shape = self.get_output_shape(a, 0);
        let node = Node::new(
            DiffOp::Add,
            vec![Port::new(shape.clone()), Port::new(shape.clone())],
            vec![Port::new(shape)],
        );
        let idx = self.diagram.add_node(node);

        // Connect inputs
        self.diagram
            .graph
            .add_edge(a, idx, compositional_core::diagram::Edge::new(0, 0));
        self.diagram
            .graph
            .add_edge(b, idx, compositional_core::diagram::Edge::new(0, 1));

        idx
    }

    /// Add a multiplication node: a * b
    pub fn mul(&mut self, a: NodeIndex, b: NodeIndex) -> NodeIndex {
        let shape = self.get_output_shape(a, 0);
        let node = Node::new(
            DiffOp::Mul,
            vec![Port::new(shape.clone()), Port::new(shape.clone())],
            vec![Port::new(shape)],
        );
        let idx = self.diagram.add_node(node);

        self.diagram
            .graph
            .add_edge(a, idx, compositional_core::diagram::Edge::new(0, 0));
        self.diagram
            .graph
            .add_edge(b, idx, compositional_core::diagram::Edge::new(0, 1));

        idx
    }

    /// Add a matrix multiplication node: A @ B
    pub fn matmul(&mut self, a: NodeIndex, b: NodeIndex) -> NodeIndex {
        let a_shape = self.get_output_shape(a, 0);
        let b_shape = self.get_output_shape(b, 0);

        // Output shape for matmul: (m, k) @ (k, n) -> (m, n)
        let out_shape = if a_shape.dims.len() == 2 && b_shape.dims.len() == 2 {
            Shape::f32_matrix(a_shape.dims[0], b_shape.dims[1])
        } else {
            a_shape.clone() // fallback
        };

        let node = Node::new(
            DiffOp::MatMul,
            vec![Port::new(a_shape), Port::new(b_shape)],
            vec![Port::new(out_shape)],
        );
        let idx = self.diagram.add_node(node);

        self.diagram
            .graph
            .add_edge(a, idx, compositional_core::diagram::Edge::new(0, 0));
        self.diagram
            .graph
            .add_edge(b, idx, compositional_core::diagram::Edge::new(0, 1));

        idx
    }

    /// Add a ReLU node.
    pub fn relu(&mut self, x: NodeIndex) -> NodeIndex {
        let shape = self.get_output_shape(x, 0);
        let node = Node::new(
            DiffOp::ReLU,
            vec![Port::new(shape.clone())],
            vec![Port::new(shape)],
        );
        let idx = self.diagram.add_node(node);

        self.diagram
            .graph
            .add_edge(x, idx, compositional_core::diagram::Edge::new(0, 0));

        idx
    }

    /// Add a sum-all node (reduce to scalar).
    pub fn sum_all(&mut self, x: NodeIndex) -> NodeIndex {
        let shape = self.get_output_shape(x, 0);
        let node = Node::new(
            DiffOp::SumAll,
            vec![Port::new(shape)],
            vec![Port::new(Shape::f32_scalar())],
        );
        let idx = self.diagram.add_node(node);

        self.diagram
            .graph
            .add_edge(x, idx, compositional_core::diagram::Edge::new(0, 0));

        idx
    }

    /// Mark a node as an output of the graph.
    pub fn mark_output(&mut self, node: NodeIndex) {
        self.output_nodes.push(node);
        self.diagram.outputs.push((node, 0));
    }

    /// Get the output shape of a node at a given port.
    fn get_output_shape(&self, node: NodeIndex, port: usize) -> Shape {
        self.diagram.graph[node].outputs[port].shape.clone()
    }

    /// Compute topological order of nodes using Kahn's algorithm.
    ///
    /// Returns nodes in an order where all dependencies are processed
    /// before the nodes that depend on them.
    pub fn topological_order(&self) -> Vec<NodeIndex> {
        let graph = &self.diagram.graph;
        let mut in_degree: HashMap<NodeIndex, usize> = HashMap::new();
        let mut queue: VecDeque<NodeIndex> = VecDeque::new();
        let mut result: Vec<NodeIndex> = Vec::new();

        // Initialize in-degrees
        for node in graph.node_indices() {
            let degree = graph.neighbors_directed(node, Direction::Incoming).count();
            in_degree.insert(node, degree);
            if degree == 0 {
                queue.push_back(node);
            }
        }

        // Process nodes
        while let Some(node) = queue.pop_front() {
            result.push(node);

            // Decrease in-degree of successors
            for successor in graph.neighbors_directed(node, Direction::Outgoing) {
                let deg = in_degree.get_mut(&successor).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue.push_back(successor);
                }
            }
        }

        result
    }

    /// Forward evaluation: compute outputs from inputs.
    ///
    /// Processes nodes in topological order, caching intermediate values.
    pub fn forward(&self, inputs: &[RTensor]) -> Vec<RTensor> {
        let graph = &self.diagram.graph;

        // Cache of computed values: node index -> output tensors
        let mut values: HashMap<NodeIndex, Vec<RTensor>> = HashMap::new();

        // Process in topological order
        for node_idx in self.topological_order() {
            let node = &graph[node_idx];

            // Gather inputs from predecessors
            let node_inputs = self.gather_inputs(node_idx, inputs, &values);

            // Execute the operation
            let outputs = node.op.forward(&node_inputs);

            values.insert(node_idx, outputs);
        }

        // Gather boundary outputs
        self.output_nodes
            .iter()
            .map(|&idx| values[&idx][0].clone())
            .collect()
    }

    /// Gather input tensors for a node from its predecessors.
    fn gather_inputs(
        &self,
        node: NodeIndex,
        boundary_inputs: &[RTensor],
        values: &HashMap<NodeIndex, Vec<RTensor>>,
    ) -> Vec<RTensor> {
        let graph = &self.diagram.graph;
        let node_data = &graph[node];

        // Check if this is an Input node
        if let DiffOp::Input { index } = node_data.op {
            return vec![boundary_inputs[index].clone()];
        }

        // Check if this is a Const node (no inputs needed)
        if let DiffOp::Const { .. } = &node_data.op {
            return vec![];
        }

        // Gather from incoming edges
        let num_inputs = node_data.inputs.len();
        let mut inputs: Vec<Option<RTensor>> = vec![None; num_inputs];

        for edge in graph.edges_directed(node, Direction::Incoming) {
            let source = edge.source();
            let edge_data = edge.weight();
            let source_outputs = &values[&source];
            let value = source_outputs[edge_data.from_port].clone();
            inputs[edge_data.to_port] = Some(value);
        }

        inputs
            .into_iter()
            .map(|x| x.expect("Missing input"))
            .collect()
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.diagram.node_count()
    }

    /// Number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.diagram.edge_count()
    }

    /// Render the graph as ASCII for debugging.
    pub fn render(&self) -> String {
        self.diagram.render_ascii()
    }
}

impl Default for DiffGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a Vec<usize> shape to a compositional_core::Shape.
fn vec_to_shape(dims: &[usize]) -> Shape {
    if dims.is_empty() {
        Shape::f32_scalar()
    } else if dims.len() == 1 {
        Shape::f32_vector(dims[0])
    } else if dims.len() == 2 {
        Shape::f32_matrix(dims[0], dims[1])
    } else {
        // General n-dimensional
        Shape::new(compositional_core::TypeId("f32"), dims.to_vec())
    }
}

// ============================================================================
// Extension: Forward eval directly on Diagram<DiffOp>
// ============================================================================

/// Extension trait to add forward evaluation to any `Diagram<DiffOp>`.
pub trait ForwardEval {
    /// Evaluate the diagram with given inputs.
    fn eval(&self, inputs: &[RTensor]) -> Vec<RTensor>;

    /// Get topological order of nodes.
    fn topo_order(&self) -> Vec<NodeIndex>;
}

impl ForwardEval for Diagram<DiffOp> {
    fn eval(&self, inputs: &[RTensor]) -> Vec<RTensor> {
        let mut values: HashMap<NodeIndex, Vec<RTensor>> = HashMap::new();

        for node_idx in self.topo_order() {
            let node = &self.graph[node_idx];

            // Gather inputs
            let node_inputs = gather_diagram_inputs(self, node_idx, inputs, &values);

            // Execute
            let outputs = node.op.forward(&node_inputs);
            values.insert(node_idx, outputs);
        }

        // Return boundary outputs
        self.outputs
            .iter()
            .map(|(idx, port)| values[idx][*port].clone())
            .collect()
    }

    fn topo_order(&self) -> Vec<NodeIndex> {
        let mut in_degree: HashMap<NodeIndex, usize> = HashMap::new();
        let mut queue: VecDeque<NodeIndex> = VecDeque::new();
        let mut result: Vec<NodeIndex> = Vec::new();

        for node in self.graph.node_indices() {
            let degree = self
                .graph
                .neighbors_directed(node, Direction::Incoming)
                .count();
            in_degree.insert(node, degree);
            if degree == 0 {
                queue.push_back(node);
            }
        }

        while let Some(node) = queue.pop_front() {
            result.push(node);
            for successor in self.graph.neighbors_directed(node, Direction::Outgoing) {
                let deg = in_degree.get_mut(&successor).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue.push_back(successor);
                }
            }
        }

        result
    }
}

fn gather_diagram_inputs(
    diagram: &Diagram<DiffOp>,
    node: NodeIndex,
    boundary_inputs: &[RTensor],
    values: &HashMap<NodeIndex, Vec<RTensor>>,
) -> Vec<RTensor> {
    let node_data = &diagram.graph[node];

    // Handle special cases
    if let DiffOp::Input { index } = node_data.op {
        return vec![boundary_inputs[index].clone()];
    }
    if let DiffOp::Const { .. } = node_data.op {
        return vec![];
    }

    // Gather from edges
    let num_inputs = node_data.inputs.len();
    let mut inputs: Vec<Option<RTensor>> = vec![None; num_inputs];

    for edge in diagram.graph.edges_directed(node, Direction::Incoming) {
        let source = edge.source();
        let edge_data = edge.weight();
        let source_outputs = &values[&source];
        let value = source_outputs[edge_data.from_port].clone();
        inputs[edge_data.to_port] = Some(value);
    }

    inputs
        .into_iter()
        .map(|x| x.expect("Missing input"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_add() {
        let mut graph = DiffGraph::new();
        let x = graph.input(0, vec![3]);
        let y = graph.input(1, vec![3]);
        let z = graph.add(x, y);
        graph.mark_output(z);

        let a = RTensor::vector(vec![1.0, 2.0, 3.0]);
        let b = RTensor::vector(vec![4.0, 5.0, 6.0]);
        let outputs = graph.forward(&[a, b]);

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_relu_chain() {
        // y = ReLU(x)
        let mut graph = DiffGraph::new();
        let x = graph.input(0, vec![4]);
        let y = graph.relu(x);
        graph.mark_output(y);

        let input = RTensor::vector(vec![-2.0, -1.0, 1.0, 2.0]);
        let outputs = graph.forward(&[input]);

        assert_eq!(outputs[0].data, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_add_relu_chain() {
        // y = ReLU(x + bias)
        let mut graph = DiffGraph::new();
        let x = graph.input(0, vec![3]);
        let bias = graph.input(1, vec![3]); // bias vector
        let sum = graph.add(x, bias);
        let y = graph.relu(sum);
        graph.mark_output(y);

        let input = RTensor::vector(vec![-2.0, 0.0, 1.0]);
        let ones = RTensor::vector(vec![1.0, 1.0, 1.0]);
        let outputs = graph.forward(&[input, ones]);

        // [-2+1, 0+1, 1+1] = [-1, 1, 2] -> ReLU -> [0, 1, 2]
        assert_eq!(outputs[0].data, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_matmul() {
        let mut graph = DiffGraph::new();
        let a = graph.input(0, vec![2, 3]);
        let b = graph.input(1, vec![3, 2]);
        let c = graph.matmul(a, b);
        graph.mark_output(c);

        // [1 2 3] × [7  8 ]   [58  64 ]
        // [4 5 6]   [9  10] = [139 154]
        //           [11 12]
        let mat_a = RTensor::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mat_b = RTensor::matrix(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let outputs = graph.forward(&[mat_a, mat_b]);

        assert_eq!(outputs[0].shape, vec![2, 2]);
        assert_eq!(outputs[0].data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_sum_all() {
        let mut graph = DiffGraph::new();
        let x = graph.input(0, vec![4]);
        let s = graph.sum_all(x);
        graph.mark_output(s);

        let input = RTensor::vector(vec![1.0, 2.0, 3.0, 4.0]);
        let outputs = graph.forward(&[input]);

        assert!(outputs[0].is_scalar());
        assert_eq!(outputs[0].as_scalar(), 10.0);
    }

    #[test]
    fn test_topological_order() {
        let mut graph = DiffGraph::new();
        let x = graph.input(0, vec![2]);
        let y = graph.input(1, vec![2]);
        let sum = graph.add(x, y);
        let result = graph.relu(sum);
        graph.mark_output(result);

        let order = graph.topological_order();

        // x and y should come before sum, sum should come before result
        let x_pos = order.iter().position(|&n| n == x).unwrap();
        let y_pos = order.iter().position(|&n| n == y).unwrap();
        let sum_pos = order.iter().position(|&n| n == sum).unwrap();
        let result_pos = order.iter().position(|&n| n == result).unwrap();

        assert!(x_pos < sum_pos);
        assert!(y_pos < sum_pos);
        assert!(sum_pos < result_pos);
    }

    #[test]
    fn test_diamond_graph() {
        // Diamond: x -> [a, b] -> c (where c = a + b)
        //          x
        //         / \
        //        a   b
        //         \ /
        //          c
        let mut graph = DiffGraph::new();
        let x = graph.input(0, vec![2]);
        let a = graph.relu(x);
        let b = graph.relu(x); // both use x - tests fan-out in DAG
        let c = graph.add(a, b); // combine outputs
        graph.mark_output(c);

        let input = RTensor::vector(vec![-1.0, 2.0]);
        let outputs = graph.forward(&[input]);

        // a = ReLU(x) = [0, 2]
        // b = ReLU(x) = [0, 2]
        // c = a + b = [0, 4]
        assert_eq!(outputs[0].data, vec![0.0, 4.0]);
    }

    #[test]
    fn test_forward_eval_trait() {
        // Test the ForwardEval trait on raw Diagram<DiffOp>
        let mut diagram: Diagram<DiffOp> = Diagram::new();

        // Create nodes
        let input_node = diagram.add_node(Node::new(
            DiffOp::Input { index: 0 },
            vec![Port::new(Shape::f32_vector(3))],
            vec![Port::new(Shape::f32_vector(3))],
        ));

        let relu_node = diagram.add_node(Node::new(
            DiffOp::ReLU,
            vec![Port::new(Shape::f32_vector(3))],
            vec![Port::new(Shape::f32_vector(3))],
        ));

        // Connect
        diagram.graph.add_edge(
            input_node,
            relu_node,
            compositional_core::diagram::Edge::new(0, 0),
        );

        // Set boundaries
        diagram.inputs = vec![(input_node, 0)];
        diagram.outputs = vec![(relu_node, 0)];

        // Evaluate using trait
        let input = RTensor::vector(vec![-1.0, 0.0, 1.0]);
        let outputs = diagram.eval(&[input]);

        assert_eq!(outputs[0].data, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_mul_operation() {
        let mut graph = DiffGraph::new();
        let x = graph.input(0, vec![3]);
        let y = graph.input(1, vec![3]);
        let z = graph.mul(x, y);
        graph.mark_output(z);

        let a = RTensor::vector(vec![1.0, 2.0, 3.0]);
        let b = RTensor::vector(vec![4.0, 5.0, 6.0]);
        let outputs = graph.forward(&[a, b]);

        assert_eq!(outputs[0].data, vec![4.0, 10.0, 18.0]);
    }
}
