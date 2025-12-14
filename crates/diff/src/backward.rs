//! # Backward Pass - Reverse-Mode Autodiff (Session 9)
//!
//! This module implements the backward pass for automatic differentiation.
//! The key insight is that the backward pass is a **functor to the opposite category**:
//!
//! - Forward: morphisms go A → B
//! - Backward: morphisms go B → A (gradients flow backward)
//! - For composition f;g, the VJP is vjp(g);vjp(f) (reverse order)
//!
//! ## Example
//!
//! ```rust
//! use compositional_diff::forward::DiffGraph;
//! use compositional_diff::backward::backward;
//! use compositional_diff::ops::RTensor;
//!
//! // Build: loss = sum(ReLU(x + y))
//! let mut graph = DiffGraph::new();
//! let x = graph.input(0, vec![3]);
//! let y = graph.input(1, vec![3]);
//! let sum = graph.add(x, y);
//! let relu = graph.relu(sum);
//! let loss = graph.sum_all(relu);
//! graph.mark_output(loss);
//!
//! // Forward pass
//! let input_x = RTensor::vector(vec![-1.0, 2.0, 3.0]);
//! let input_y = RTensor::vector(vec![2.0, -1.0, 1.0]);
//! let (outputs, forward_cache) = graph.forward_with_cache(&[input_x.clone(), input_y.clone()]);
//!
//! // Backward pass (seed gradient = 1.0 for scalar loss)
//! let grads = backward(&graph, &forward_cache, RTensor::scalar(1.0));
//!
//! // grads contains ∂loss/∂x and ∂loss/∂y
//! ```

use crate::forward::DiffGraph;
use crate::ops::{DiffOp, RTensor};
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use std::collections::HashMap;

/// Cache of forward pass values needed for backward pass.
pub type ForwardCache = HashMap<NodeIndex, ForwardValue>;

/// Cached values from forward pass for a single node.
#[derive(Clone)]
pub struct ForwardValue {
    /// Inputs to this node
    pub inputs: Vec<RTensor>,
    /// Outputs from this node
    pub outputs: Vec<RTensor>,
}

/// Perform backward pass to compute gradients.
///
/// # Arguments
///
/// * `graph` - The computation graph
/// * `forward_cache` - Cached values from forward pass
/// * `seed_grad` - Initial gradient (typically 1.0 for scalar loss)
///
/// # Returns
///
/// HashMap mapping node indices to their output gradients.
pub fn backward(
    graph: &DiffGraph,
    forward_cache: &ForwardCache,
    seed_grad: RTensor,
) -> HashMap<NodeIndex, Vec<RTensor>> {
    let dag = &graph.diagram.graph;

    // Gradients for each node's outputs
    let mut grads: HashMap<NodeIndex, Vec<RTensor>> = HashMap::new();

    // Initialize output node gradient with seed
    if let Some(&output_node) = graph.output_nodes.first() {
        grads.insert(output_node, vec![seed_grad]);
    }

    // Process in REVERSE topological order
    let topo_order = graph.topological_order();
    for node_idx in topo_order.into_iter().rev() {
        // Skip if no gradient has reached this node yet
        if !grads.contains_key(&node_idx) {
            continue;
        }

        let node = &dag[node_idx];
        let output_grads = &grads[&node_idx];
        let forward_val = &forward_cache[&node_idx];

        // Compute VJP: output gradients → input gradients
        let input_grads = node.op.vjp(&forward_val.inputs, output_grads);

        // Propagate gradients to predecessor nodes
        for edge in dag.edges_directed(node_idx, Direction::Incoming) {
            let source = edge.source();
            let edge_data = edge.weight();

            // Use to_port to select the correct input gradient
            // (edge_data.to_port is which input slot of this node the edge connects to)
            if edge_data.to_port < input_grads.len() {
                let grad = input_grads[edge_data.to_port].clone();

                // Accumulate gradient (for nodes with multiple consumers)
                let node_grads = grads.entry(source).or_insert_with(|| {
                    // Initialize with zeros matching the output shape
                    let source_outputs = &forward_cache[&source].outputs;
                    source_outputs.iter().map(|t| t.zeros_like()).collect()
                });

                if let Some(existing) = node_grads.get_mut(edge_data.from_port) {
                    *existing = existing.add(&grad);
                }
            }
        }
    }

    grads
}

/// Compute gradient of a scalar loss with respect to inputs.
///
/// This is the main entry point for gradient computation.
///
/// # Arguments
///
/// * `graph` - The computation graph (must have scalar output)
/// * `inputs` - Input tensors
///
/// # Returns
///
/// Gradients with respect to each input tensor.
pub fn grad(graph: &DiffGraph, inputs: &[RTensor]) -> Vec<RTensor> {
    // Forward pass with caching
    let (_, forward_cache) = graph.forward_with_cache(inputs);

    // Backward pass with seed = 1.0
    let grads = backward(graph, &forward_cache, RTensor::scalar(1.0));

    // Extract gradients for input nodes (in order)
    let mut input_grads = Vec::new();
    for (i, input) in inputs.iter().enumerate() {
        if let Some(&node_idx) = graph.input_nodes.get(&i) {
            if let Some(node_grads) = grads.get(&node_idx) {
                input_grads.push(node_grads[0].clone());
            } else {
                // No gradient reached this input
                input_grads.push(input.zeros_like());
            }
        }
    }

    input_grads
}

/// Numerical gradient computation for testing.
///
/// Uses central differences: (f(x+h) - f(x-h)) / 2h
pub fn numerical_gradient(
    graph: &DiffGraph,
    inputs: &[RTensor],
    input_idx: usize,
    elem_idx: usize,
    h: f32,
) -> f32 {
    let mut inputs_plus = inputs.to_vec();
    let mut inputs_minus = inputs.to_vec();

    inputs_plus[input_idx].data[elem_idx] += h;
    inputs_minus[input_idx].data[elem_idx] -= h;

    let f_plus = graph.forward(&inputs_plus)[0].as_scalar();
    let f_minus = graph.forward(&inputs_minus)[0].as_scalar();

    (f_plus - f_minus) / (2.0 * h)
}

/// Check analytical gradients against numerical gradients.
///
/// # Arguments
///
/// * `graph` - The computation graph
/// * `inputs` - Input tensors
/// * `h` - Step size for numerical differentiation (e.g., 1e-4)
/// * `tolerance` - Maximum allowed difference (e.g., 1e-4)
///
/// # Returns
///
/// True if all gradients match within tolerance.
pub fn grad_check(
    graph: &DiffGraph,
    inputs: &[RTensor],
    h: f32,
    tolerance: f32,
) -> Result<(), GradCheckError> {
    let analytical_grads = grad(graph, inputs);

    for (input_idx, analytical) in analytical_grads.iter().enumerate() {
        for elem_idx in 0..analytical.data.len() {
            let numerical = numerical_gradient(graph, inputs, input_idx, elem_idx, h);
            let analytical_val = analytical.data[elem_idx];
            let diff = (numerical - analytical_val).abs();

            // Use relative error for large values
            let scale = analytical_val.abs().max(numerical.abs()).max(1.0);
            let rel_diff = diff / scale;

            if rel_diff > tolerance && diff > tolerance {
                return Err(GradCheckError {
                    input_idx,
                    elem_idx,
                    analytical: analytical_val,
                    numerical,
                    diff,
                });
            }
        }
    }

    Ok(())
}

/// Error from gradient checking.
#[derive(Debug)]
pub struct GradCheckError {
    pub input_idx: usize,
    pub elem_idx: usize,
    pub analytical: f32,
    pub numerical: f32,
    pub diff: f32,
}

impl std::fmt::Display for GradCheckError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Gradient mismatch at input[{}][{}]: analytical={}, numerical={}, diff={}",
            self.input_idx, self.elem_idx, self.analytical, self.numerical, self.diff
        )
    }
}

impl std::error::Error for GradCheckError {}

// ============================================================================
// Extension to DiffGraph for forward with cache
// ============================================================================

impl DiffGraph {
    /// Forward pass that returns cached values needed for backward.
    pub fn forward_with_cache(&self, inputs: &[RTensor]) -> (Vec<RTensor>, ForwardCache) {
        let graph = &self.diagram.graph;
        let mut cache: ForwardCache = HashMap::new();

        // Process in topological order
        for node_idx in self.topological_order() {
            let node = &graph[node_idx];

            // Gather inputs from predecessors
            let node_inputs = self.gather_inputs_for_backward(node_idx, inputs, &cache);

            // Execute the operation
            let outputs = node.op.forward(&node_inputs);

            // Cache both inputs and outputs
            cache.insert(
                node_idx,
                ForwardValue {
                    inputs: node_inputs,
                    outputs: outputs.clone(),
                },
            );
        }

        // Gather boundary outputs
        let result = self
            .output_nodes
            .iter()
            .map(|&idx| cache[&idx].outputs[0].clone())
            .collect();

        (result, cache)
    }

    /// Gather inputs for a node (similar to forward, but stores them for backward).
    fn gather_inputs_for_backward(
        &self,
        node: NodeIndex,
        boundary_inputs: &[RTensor],
        cache: &ForwardCache,
    ) -> Vec<RTensor> {
        let graph = &self.diagram.graph;
        let node_data = &graph[node];

        // Check if this is an Input node
        if let DiffOp::Input { index } = node_data.op {
            return vec![boundary_inputs[index].clone()];
        }

        // Check if this is a Const node
        if let DiffOp::Const { .. } = node_data.op {
            return vec![];
        }

        // Gather from incoming edges
        let num_inputs = node_data.inputs.len();
        let mut inputs: Vec<Option<RTensor>> = vec![None; num_inputs];

        for edge in graph.edges_directed(node, Direction::Incoming) {
            let source = edge.source();
            let edge_data = edge.weight();
            let source_outputs = &cache[&source].outputs;
            let value = source_outputs[edge_data.from_port].clone();
            inputs[edge_data.to_port] = Some(value);
        }

        inputs
            .into_iter()
            .map(|x| x.expect("Missing input"))
            .collect()
    }

    /// Access to input_nodes for backward pass.
    pub fn get_input_nodes(&self) -> &HashMap<usize, NodeIndex> {
        &self.input_nodes
    }

    /// Access to output_nodes for backward pass.
    pub fn get_output_nodes(&self) -> &[NodeIndex] {
        &self.output_nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backward_simple_add() {
        // loss = sum(x + y)
        let mut graph = DiffGraph::new();
        let x = graph.input(0, vec![3]);
        let y = graph.input(1, vec![3]);
        let sum = graph.add(x, y);
        let loss = graph.sum_all(sum);
        graph.mark_output(loss);

        let input_x = RTensor::vector(vec![1.0, 2.0, 3.0]);
        let input_y = RTensor::vector(vec![4.0, 5.0, 6.0]);

        let grads = grad(&graph, &[input_x, input_y]);

        // ∂loss/∂x = [1, 1, 1], ∂loss/∂y = [1, 1, 1]
        assert_eq!(grads[0].data, vec![1.0, 1.0, 1.0]);
        assert_eq!(grads[1].data, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_backward_mul() {
        // loss = sum(x * y)
        let mut graph = DiffGraph::new();
        let x = graph.input(0, vec![3]);
        let y = graph.input(1, vec![3]);
        let prod = graph.mul(x, y);
        let loss = graph.sum_all(prod);
        graph.mark_output(loss);

        let input_x = RTensor::vector(vec![1.0, 2.0, 3.0]);
        let input_y = RTensor::vector(vec![4.0, 5.0, 6.0]);

        let grads = grad(&graph, &[input_x, input_y]);

        // ∂loss/∂x = y, ∂loss/∂y = x
        assert_eq!(grads[0].data, vec![4.0, 5.0, 6.0]);
        assert_eq!(grads[1].data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_backward_relu() {
        // loss = sum(ReLU(x))
        let mut graph = DiffGraph::new();
        let x = graph.input(0, vec![4]);
        let relu = graph.relu(x);
        let loss = graph.sum_all(relu);
        graph.mark_output(loss);

        let input_x = RTensor::vector(vec![-1.0, 0.0, 1.0, 2.0]);
        let grads = grad(&graph, &[input_x]);

        // ∂loss/∂x = [0, 0, 1, 1] (gradient blocked where input <= 0)
        assert_eq!(grads[0].data, vec![0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_backward_chain() {
        // loss = sum(ReLU(x + y))
        let mut graph = DiffGraph::new();
        let x = graph.input(0, vec![3]);
        let y = graph.input(1, vec![3]);
        let sum = graph.add(x, y);
        let relu = graph.relu(sum);
        let loss = graph.sum_all(relu);
        graph.mark_output(loss);

        let input_x = RTensor::vector(vec![-2.0, 0.0, 2.0]);
        let input_y = RTensor::vector(vec![1.0, 1.0, 1.0]);

        // x + y = [-1, 1, 3]
        // ReLU(x + y) = [0, 1, 3]
        // loss = 4

        let grads = grad(&graph, &[input_x, input_y]);

        // ∂loss/∂(x+y) = [0, 1, 1] (blocked where x+y <= 0)
        // ∂loss/∂x = ∂loss/∂y = [0, 1, 1]
        assert_eq!(grads[0].data, vec![0.0, 1.0, 1.0]);
        assert_eq!(grads[1].data, vec![0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_backward_matmul() {
        // loss = sum(A @ B)
        let mut graph = DiffGraph::new();
        let a = graph.input(0, vec![2, 3]);
        let b = graph.input(1, vec![3, 2]);
        let c = graph.matmul(a, b);
        let loss = graph.sum_all(c);
        graph.mark_output(loss);

        // A = [[1, 2, 3], [4, 5, 6]]
        // B = [[1, 2], [3, 4], [5, 6]]
        let input_a = RTensor::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let input_b = RTensor::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let grads = grad(&graph, &[input_a, input_b]);

        // ∂loss/∂A = ones(2,2) @ Bᵀ = [[3, 7, 11], [3, 7, 11]]
        // ∂loss/∂B = Aᵀ @ ones(2,2) = [[5, 5], [7, 7], [9, 9]]
        assert_eq!(grads[0].shape, vec![2, 3]);
        assert_eq!(grads[1].shape, vec![3, 2]);
    }

    #[test]
    fn test_grad_check_add() {
        let mut graph = DiffGraph::new();
        let x = graph.input(0, vec![3]);
        let y = graph.input(1, vec![3]);
        let sum = graph.add(x, y);
        let loss = graph.sum_all(sum);
        graph.mark_output(loss);

        let input_x = RTensor::vector(vec![1.0, 2.0, 3.0]);
        let input_y = RTensor::vector(vec![4.0, 5.0, 6.0]);

        // Note: Using h=1e-2 and tolerance=1e-2 because f32 precision limits
        // the accuracy of numerical gradients computed with smaller h values.
        let result = grad_check(&graph, &[input_x, input_y], 1e-2, 1e-2);
        assert!(result.is_ok(), "Grad check failed: {:?}", result);
    }

    #[test]
    fn test_grad_check_mul() {
        let mut graph = DiffGraph::new();
        let x = graph.input(0, vec![3]);
        let y = graph.input(1, vec![3]);
        let prod = graph.mul(x, y);
        let loss = graph.sum_all(prod);
        graph.mark_output(loss);

        let input_x = RTensor::vector(vec![1.0, 2.0, 3.0]);
        let input_y = RTensor::vector(vec![4.0, 5.0, 6.0]);

        // f32 precision limits numerical gradient accuracy
        let result = grad_check(&graph, &[input_x, input_y], 1e-2, 1e-2);
        assert!(result.is_ok(), "Grad check failed: {:?}", result);
    }

    #[test]
    fn test_grad_check_relu() {
        let mut graph = DiffGraph::new();
        let x = graph.input(0, vec![4]);
        let relu = graph.relu(x);
        let loss = graph.sum_all(relu);
        graph.mark_output(loss);

        // Avoid testing exactly at 0 where ReLU is non-differentiable
        let input_x = RTensor::vector(vec![-1.0, -0.5, 0.5, 1.0]);

        // f32 precision limits numerical gradient accuracy
        let result = grad_check(&graph, &[input_x], 1e-2, 1e-2);
        assert!(result.is_ok(), "Grad check failed: {:?}", result);
    }

    #[test]
    fn test_grad_check_matmul() {
        let mut graph = DiffGraph::new();
        let a = graph.input(0, vec![2, 2]);
        let b = graph.input(1, vec![2, 2]);
        let c = graph.matmul(a, b);
        let loss = graph.sum_all(c);
        graph.mark_output(loss);

        let input_a = RTensor::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let input_b = RTensor::matrix(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

        // f32 precision limits numerical gradient accuracy
        let result = grad_check(&graph, &[input_a, input_b], 1e-2, 1e-2);
        assert!(result.is_ok(), "Grad check failed: {:?}", result);
    }

    #[test]
    fn test_grad_check_chain() {
        let mut graph = DiffGraph::new();
        let x = graph.input(0, vec![3]);
        let y = graph.input(1, vec![3]);
        let sum = graph.add(x, y);
        let relu = graph.relu(sum);
        let loss = graph.sum_all(relu);
        graph.mark_output(loss);

        // Avoid ReLU boundary
        let input_x = RTensor::vector(vec![-2.0, 0.5, 2.0]);
        let input_y = RTensor::vector(vec![1.0, 1.0, 1.0]);

        // f32 precision limits numerical gradient accuracy
        let result = grad_check(&graph, &[input_x, input_y], 1e-2, 1e-2);
        assert!(result.is_ok(), "Grad check failed: {:?}", result);
    }
}
