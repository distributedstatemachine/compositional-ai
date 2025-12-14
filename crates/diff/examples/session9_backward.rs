//! Session 9: Reverse-Mode Autodiff (VJP Rules)
//!
//! Run with: cargo run -p compositional-diff --example session9_backward
//!
//! This example demonstrates:
//! - VJP (Vector-Jacobian Product) for each operation
//! - Backward pass through computation graphs
//! - Gradient accumulation for nodes with multiple consumers
//! - Numerical gradient checking for verification
//!
//! Key insight: The backward pass is a functor to the opposite category.
//! Forward composition f;g becomes backward composition vjp(g);vjp(f).

use compositional_diff::backward::{grad, grad_check, numerical_gradient};
use compositional_diff::forward::DiffGraph;
use compositional_diff::ops::RTensor;

fn main() {
    println!("=== Session 9: Reverse-Mode Autodiff ===\n");

    // -------------------------------------------------------------------------
    // 1. Simple Addition: VJP distributes gradient to both inputs
    // -------------------------------------------------------------------------
    println!("1. Addition VJP: z = x + y");
    println!("--------------------------");
    println!();
    println!("Forward:  z = x + y");
    println!("VJP:      grad_x = grad_z, grad_y = grad_z");
    println!();

    let mut graph = DiffGraph::new();
    let x = graph.input(0, vec![3]);
    let y = graph.input(1, vec![3]);
    let sum = graph.add(x, y);
    let loss = graph.sum_all(sum);
    graph.mark_output(loss);

    let input_x = RTensor::vector(vec![1.0, 2.0, 3.0]);
    let input_y = RTensor::vector(vec![4.0, 5.0, 6.0]);

    let grads = grad(&graph, &[input_x.clone(), input_y.clone()]);
    println!("x = [1, 2, 3]");
    println!("y = [4, 5, 6]");
    println!("loss = sum(x + y) = sum([5, 7, 9]) = 21");
    println!();
    println!("Gradients:");
    println!("  grad_x = {:?}", grads[0].data);
    println!("  grad_y = {:?}", grads[1].data);
    println!("  (Both are [1, 1, 1] because each element equally affects the sum)");
    println!();

    // -------------------------------------------------------------------------
    // 2. Multiplication: VJP swaps and scales
    // -------------------------------------------------------------------------
    println!("2. Multiplication VJP: z = x * y");
    println!("--------------------------------");
    println!();
    println!("Forward:  z = x * y (element-wise)");
    println!("VJP:      grad_x = grad_z * y, grad_y = grad_z * x");
    println!();

    let mut graph = DiffGraph::new();
    let x = graph.input(0, vec![3]);
    let y = graph.input(1, vec![3]);
    let prod = graph.mul(x, y);
    let loss = graph.sum_all(prod);
    graph.mark_output(loss);

    let grads = grad(&graph, &[input_x.clone(), input_y.clone()]);
    println!("x = [1, 2, 3]");
    println!("y = [4, 5, 6]");
    println!("loss = sum(x * y) = sum([4, 10, 18]) = 32");
    println!();
    println!("Gradients:");
    println!("  grad_x = {:?} (= y)", grads[0].data);
    println!("  grad_y = {:?} (= x)", grads[1].data);
    println!();

    // -------------------------------------------------------------------------
    // 3. ReLU: Gradient gate
    // -------------------------------------------------------------------------
    println!("3. ReLU VJP: y = max(0, x)");
    println!("--------------------------");
    println!();
    println!("Forward:  y = max(0, x)");
    println!("VJP:      grad_x = grad_y * (x > 0 ? 1 : 0)");
    println!();

    let mut graph = DiffGraph::new();
    let x = graph.input(0, vec![5]);
    let relu = graph.relu(x);
    let loss = graph.sum_all(relu);
    graph.mark_output(loss);

    let input = RTensor::vector(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    let grads = grad(&graph, &[input]);

    println!("x = [-2, -1, 0, 1, 2]");
    println!("ReLU(x) = [0, 0, 0, 1, 2]");
    println!("loss = sum(ReLU(x)) = 3");
    println!();
    println!("Gradients:");
    println!("  grad_x = {:?}", grads[0].data);
    println!("  (Gradient blocked where x <= 0, passes through where x > 0)");
    println!();

    // -------------------------------------------------------------------------
    // 4. Chained Operations: Chain Rule in Action
    // -------------------------------------------------------------------------
    println!("4. Chain Rule: loss = sum(ReLU(x + y))");
    println!("--------------------------------------");
    println!();
    println!("Forward:  t = x + y, u = ReLU(t), loss = sum(u)");
    println!("Backward: grad_t = grad_u * (t > 0), grad_x = grad_y = grad_t");
    println!();

    let mut graph = DiffGraph::new();
    let x = graph.input(0, vec![4]);
    let y = graph.input(1, vec![4]);
    let sum = graph.add(x, y);
    let relu = graph.relu(sum);
    let loss = graph.sum_all(relu);
    graph.mark_output(loss);

    let input_x = RTensor::vector(vec![-3.0, -1.0, 1.0, 3.0]);
    let input_y = RTensor::vector(vec![1.0, 1.0, 1.0, 1.0]);
    let grads = grad(&graph, &[input_x, input_y]);

    println!("x = [-3, -1, 1, 3]");
    println!("y = [1, 1, 1, 1]");
    println!("x + y = [-2, 0, 2, 4]");
    println!("ReLU(x + y) = [0, 0, 2, 4]");
    println!("loss = 6");
    println!();
    println!("Gradients:");
    println!("  grad_x = {:?}", grads[0].data);
    println!("  grad_y = {:?}", grads[1].data);
    println!("  (Gradient is [0, 0, 1, 1] because ReLU blocks at x+y <= 0)");
    println!();

    // -------------------------------------------------------------------------
    // 5. Matrix Multiplication VJP
    // -------------------------------------------------------------------------
    println!("5. MatMul VJP: C = A @ B");
    println!("------------------------");
    println!();
    println!("Forward:  C = A @ B  where A:(m,k), B:(k,n), C:(m,n)");
    println!("VJP:      grad_A = grad_C @ B^T");
    println!("          grad_B = A^T @ grad_C");
    println!();

    let mut graph = DiffGraph::new();
    let a = graph.input(0, vec![2, 3]);
    let b = graph.input(1, vec![3, 2]);
    let c = graph.matmul(a, b);
    let loss = graph.sum_all(c);
    graph.mark_output(loss);

    let input_a = RTensor::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let input_b = RTensor::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let grads = grad(&graph, &[input_a, input_b]);

    println!("A = [[1, 2, 3],");
    println!("     [4, 5, 6]]  (2x3)");
    println!();
    println!("B = [[1, 2],");
    println!("     [3, 4],");
    println!("     [5, 6]]  (3x2)");
    println!();
    println!("C = A @ B = [[22, 28],");
    println!("             [49, 64]]  (2x2)");
    println!();
    println!("loss = sum(C) = 163");
    println!();
    println!("Gradients:");
    println!(
        "  grad_A shape: {:?}, data: {:?}",
        grads[0].shape, grads[0].data
    );
    println!("    (= ones(2,2) @ B^T, shape 2x3)");
    println!(
        "  grad_B shape: {:?}, data: {:?}",
        grads[1].shape, grads[1].data
    );
    println!("    (= A^T @ ones(2,2), shape 3x2)");
    println!();

    // -------------------------------------------------------------------------
    // 6. Numerical Gradient Verification
    // -------------------------------------------------------------------------
    println!("6. Numerical Gradient Verification");
    println!("-----------------------------------");
    println!();
    println!("We verify analytical gradients using central differences:");
    println!("  numerical_grad[i] = (f(x+h) - f(x-h)) / 2h");
    println!();

    let mut graph = DiffGraph::new();
    let x = graph.input(0, vec![3]);
    let y = graph.input(1, vec![3]);
    let prod = graph.mul(x, y);
    let loss = graph.sum_all(prod);
    graph.mark_output(loss);

    let input_x = RTensor::vector(vec![1.0, 2.0, 3.0]);
    let input_y = RTensor::vector(vec![4.0, 5.0, 6.0]);

    let h = 1e-2;
    println!("For loss = sum(x * y), checking grad_x[0]:");
    println!();
    let analytical = grad(&graph, &[input_x.clone(), input_y.clone()])[0].data[0];
    let numerical = numerical_gradient(&graph, &[input_x.clone(), input_y.clone()], 0, 0, h);
    println!("  Analytical: {:.6}", analytical);
    println!("  Numerical:  {:.6}", numerical);
    println!("  Difference: {:.6}", (analytical - numerical).abs());
    println!();

    // Full gradient check
    let result = grad_check(&graph, &[input_x, input_y], 1e-2, 1e-2);
    match result {
        Ok(()) => println!("  Gradient check PASSED!"),
        Err(e) => println!("  Gradient check failed: {}", e),
    }
    println!();

    // -------------------------------------------------------------------------
    // 7. The Opposite Category View
    // -------------------------------------------------------------------------
    println!("7. The Categorical View");
    println!("-----------------------");
    println!();
    println!("The backward pass is a functor to the opposite category:");
    println!();
    println!("Forward direction:");
    println!("  x ──[+]──▶ t ──[ReLU]──▶ u ──[sum]──▶ loss");
    println!("  y ──┘");
    println!();
    println!("Backward direction (opposite category):");
    println!("  grad_x ◀──[vjp(+)]── grad_t ◀──[vjp(ReLU)]── grad_u ◀──[vjp(sum)]── 1.0");
    println!("  grad_y ◀──┘");
    println!();
    println!("Key properties:");
    println!("  • Forward composition f;g becomes backward composition vjp(g);vjp(f)");
    println!("  • Gradients flow backward, values flow forward");
    println!("  • The functor preserves the compositional structure");
    println!();

    // -------------------------------------------------------------------------
    // 8. VJP Rules Summary
    // -------------------------------------------------------------------------
    println!("8. VJP Rules Summary");
    println!("--------------------");
    println!();
    println!("| Operation | Forward      | VJP (backward)                    |");
    println!("|-----------|--------------|-----------------------------------|");
    println!("| Add       | z = x + y    | grad_x = grad_z, grad_y = grad_z  |");
    println!("| Mul       | z = x * y    | grad_x = grad_z*y, grad_y = grad_z*x |");
    println!("| ReLU      | y = max(0,x) | grad_x = grad_y * (x > 0)         |");
    println!("| MatMul    | C = A @ B    | grad_A = grad_C @ B^T             |");
    println!("|           |              | grad_B = A^T @ grad_C             |");
    println!("| SumAll    | s = sum(x)   | grad_x = broadcast(grad_s)        |");
    println!("| Copy      | (y,y) = (x,x)| grad_x = sum(grad_y1, grad_y2)    |");
    println!();

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    println!("=== Session 9 Complete ===");
    println!();
    println!("We implemented reverse-mode autodiff via VJP rules:");
    println!("  • VJP: Given output gradients, compute input gradients");
    println!("  • Backward pass: Reverse topological order, accumulate gradients");
    println!("  • Categorical insight: Backward is a functor to Op(C)");
    println!("  • Verification: Numerical gradient checking confirms correctness");
    println!();
    println!("Next: Session 10 - Parameters and Training");
}
