//! Session 8: Forward Evaluation
//!
//! Run with: cargo run -p compositional-diff --example session8_forward
//!
//! This example demonstrates:
//! - Building computation graphs with DiffGraph
//! - Topological sort for execution order
//! - Forward evaluation (deterministic semantics)
//! - Various operations: Add, MatMul, ReLU, SumAll

use compositional_diff::forward::DiffGraph;
use compositional_diff::ops::RTensor;

fn main() {
    println!("=== Session 8: Forward Evaluation ===\n");

    // -------------------------------------------------------------------------
    // Simple Addition
    // -------------------------------------------------------------------------
    println!("1. Simple Addition: z = x + y");
    println!("-----------------------------");

    let mut graph = DiffGraph::new();
    let x = graph.input(0, vec![3]);
    let y = graph.input(1, vec![3]);
    let z = graph.add(x, y);
    graph.mark_output(z);

    let input_x = RTensor::vector(vec![1.0, 2.0, 3.0]);
    let input_y = RTensor::vector(vec![4.0, 5.0, 6.0]);
    let outputs = graph.forward(&[input_x, input_y]);

    println!("x = [1, 2, 3]");
    println!("y = [4, 5, 6]");
    println!("z = x + y = {:?}", outputs[0].data);
    println!();

    // -------------------------------------------------------------------------
    // ReLU Activation
    // -------------------------------------------------------------------------
    println!("2. ReLU Activation: y = ReLU(x)");
    println!("-------------------------------");

    let mut graph = DiffGraph::new();
    let x = graph.input(0, vec![5]);
    let y = graph.relu(x);
    graph.mark_output(y);

    let input = RTensor::vector(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    let outputs = graph.forward(&[input]);

    println!("x = [-2, -1, 0, 1, 2]");
    println!("ReLU(x) = {:?}", outputs[0].data);
    println!("  (negative values become 0)");
    println!();

    // -------------------------------------------------------------------------
    // Chained Operations: z = ReLU(x + y)
    // -------------------------------------------------------------------------
    println!("3. Chained Operations: z = ReLU(x + y)");
    println!("--------------------------------------");

    let mut graph = DiffGraph::new();
    let x = graph.input(0, vec![4]);
    let y = graph.input(1, vec![4]);
    let sum = graph.add(x, y);
    let z = graph.relu(sum);
    graph.mark_output(z);

    let input_x = RTensor::vector(vec![-3.0, -1.0, 1.0, 3.0]);
    let input_y = RTensor::vector(vec![1.0, 1.0, 1.0, 1.0]);
    let outputs = graph.forward(&[input_x, input_y]);

    println!("x = [-3, -1, 1, 3]");
    println!("y = [1, 1, 1, 1]");
    println!("x + y = [-2, 0, 2, 4]");
    println!("ReLU(x + y) = {:?}", outputs[0].data);
    println!();

    // -------------------------------------------------------------------------
    // Matrix Multiplication
    // -------------------------------------------------------------------------
    println!("4. Matrix Multiplication: C = A @ B");
    println!("-----------------------------------");

    let mut graph = DiffGraph::new();
    let a = graph.input(0, vec![2, 3]);
    let b = graph.input(1, vec![3, 2]);
    let c = graph.matmul(a, b);
    graph.mark_output(c);

    // A = [1 2 3]    B = [7  8 ]
    //     [4 5 6]        [9  10]
    //                    [11 12]
    let mat_a = RTensor::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mat_b = RTensor::matrix(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let outputs = graph.forward(&[mat_a, mat_b]);

    println!("A = [[1, 2, 3],");
    println!("     [4, 5, 6]]");
    println!();
    println!("B = [[7,  8 ],");
    println!("     [9,  10],");
    println!("     [11, 12]]");
    println!();
    println!("C = A @ B = {:?}", outputs[0].data);
    println!("Shape: {:?}", outputs[0].shape);
    println!("  (2×3) @ (3×2) = (2×2)");
    println!();

    // -------------------------------------------------------------------------
    // Sum All (Reduction to Scalar)
    // -------------------------------------------------------------------------
    println!("5. Sum All: s = sum(x)");
    println!("----------------------");

    let mut graph = DiffGraph::new();
    let x = graph.input(0, vec![4]);
    let s = graph.sum_all(x);
    graph.mark_output(s);

    let input = RTensor::vector(vec![1.0, 2.0, 3.0, 4.0]);
    let outputs = graph.forward(&[input]);

    println!("x = [1, 2, 3, 4]");
    println!("sum(x) = {}", outputs[0].as_scalar());
    println!();

    // -------------------------------------------------------------------------
    // Neural Network Layer: y = ReLU(W @ x + b)
    // -------------------------------------------------------------------------
    println!("6. Neural Network Layer: y = ReLU(W @ x)");
    println!("----------------------------------------");

    let mut graph = DiffGraph::new();
    let w = graph.input(0, vec![2, 3]); // weights: 2×3
    let x = graph.input(1, vec![3, 1]); // input: 3×1 column vector
    let wx = graph.matmul(w, x); // W @ x: 2×1
    let y = graph.relu(wx); // ReLU(W @ x)
    graph.mark_output(y);

    // W = [[0.5, -0.5, 0.1],
    //      [-0.3, 0.8, 0.2]]
    let weights = RTensor::matrix(2, 3, vec![0.5, -0.5, 0.1, -0.3, 0.8, 0.2]);
    // x = [1, 2, 3]ᵀ
    let input = RTensor::matrix(3, 1, vec![1.0, 2.0, 3.0]);
    let outputs = graph.forward(&[weights, input]);

    println!("W = [[0.5, -0.5, 0.1],");
    println!("     [-0.3, 0.8, 0.2]]");
    println!();
    println!("x = [1, 2, 3]ᵀ");
    println!();
    println!("W @ x = [0.5×1 - 0.5×2 + 0.1×3, -0.3×1 + 0.8×2 + 0.2×3]ᵀ");
    println!("      = [-0.2, 1.9]ᵀ");
    println!();
    println!("ReLU(W @ x) = {:?}", outputs[0].data);
    println!();

    // -------------------------------------------------------------------------
    // Topological Order Visualization
    // -------------------------------------------------------------------------
    println!("7. Topological Order");
    println!("--------------------");

    let mut graph = DiffGraph::new();
    let x = graph.input(0, vec![2]);
    let y = graph.input(1, vec![2]);
    let sum = graph.add(x, y);
    let prod = graph.mul(x, y);
    let combined = graph.add(sum, prod);
    let result = graph.relu(combined);
    graph.mark_output(result);

    println!("Graph: result = ReLU((x + y) + (x * y))");
    println!();

    let order = graph.topological_order();
    println!("Topological order (node indices):");
    for (i, node_idx) in order.iter().enumerate() {
        let node = &graph.diagram.graph[*node_idx];
        println!("  Step {}: {:?}", i + 1, node.op);
    }
    println!();
    println!("This order ensures all dependencies are computed first.");
    println!();

    // -------------------------------------------------------------------------
    // Computation Graph as String Diagram
    // -------------------------------------------------------------------------
    println!("8. Computation Graph Structure");
    println!("------------------------------");
    println!();
    println!("A computation graph IS a string diagram:");
    println!();
    println!("        ┌─────┐");
    println!("   x ───│  W  │───┐");
    println!("        └─────┘   │   ┌─────┐    ┌──────┐");
    println!("                  ├───│  +  │────│ ReLU │──── y");
    println!("        ┌─────┐   │   └─────┘    └──────┘");
    println!("   b ───│     │───┘");
    println!("        └─────┘");
    println!();
    println!("Forward evaluation is a functor: Diagram → Value");
    println!("  - Preserves sequential composition: eval(f ; g) = eval(g)(eval(f)(x))");
    println!("  - Preserves parallel composition: eval(f ⊗ g) = (eval(f), eval(g))");
    println!();

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    println!("9. Summary");
    println!("----------");
    println!();
    println!("Session 8 introduced forward evaluation:");
    println!();
    println!("  • DiffOp: Add, Mul, MatMul, ReLU, SumAll, etc.");
    println!("  • RTensor: Runtime tensor with dynamic shapes");
    println!("  • DiffGraph: Builder for computation graphs");
    println!("  • Topological sort: Kahn's algorithm for execution order");
    println!("  • Forward pass: Deterministic (diagram + inputs → outputs)");
    println!();
    println!("Key insight: Forward evaluation processes nodes in topological");
    println!("order, caching intermediate values. This is the 'fold' over the");
    println!("computation graph viewed as a string diagram.");
    println!();

    println!("=== Session 8 Complete ===");
}
