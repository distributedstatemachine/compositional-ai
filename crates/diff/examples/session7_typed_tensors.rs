//! Session 7: Typed Tensors - Compile-Time Dimension Checking
//!
//! Run with: cargo run -p compositional-diff --example session7_typed_tensors
//!
//! This example demonstrates:
//! - Const generic tensors with compile-time shape checking
//! - Type-safe matrix multiplication
//! - Neural network layers with dimension guarantees
//! - Categorical composition (sequential and parallel)

use compositional_diff::tensor::{compose, tensor_product, Linear, Tensor, MLP};

fn main() {
    println!("=== Session 7: Typed Tensors ===\n");

    // -------------------------------------------------------------------------
    // Basic Tensor Operations
    // -------------------------------------------------------------------------
    println!("1. Basic Tensor Operations");
    println!("--------------------------");

    let a: Tensor<f32, 2, 3> = Tensor::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    println!("Tensor a (2×3):");
    println!("  Type: Tensor<f32, 2, 3>");
    println!("  Data: {:?}", a.data());
    println!();

    // -------------------------------------------------------------------------
    // Compile-Time Shape Checking
    // -------------------------------------------------------------------------
    println!("2. Compile-Time Shape Checking");
    println!("------------------------------");

    // Matrix multiplication: (2×3) × (3×4) = (2×4)
    let b: Tensor<f32, 3, 4> = Tensor::zeros();
    let c: Tensor<f32, 2, 4> = a.matmul(&b);

    println!("a: Tensor<f32, 2, 3>  (2×3 matrix)");
    println!("b: Tensor<f32, 3, 4>  (3×4 matrix)");
    println!("c = a.matmul(&b): Tensor<f32, 2, 4>  (2×4 matrix)");
    println!();
    println!("The inner dimensions (3) must match - enforced by type system!");
    println!();

    // This would NOT compile:
    // let bad: Tensor<f32, 5, 4> = Tensor::zeros();
    // let _ = a.matmul(&bad);  // Error: expected Tensor<_, 3, 4>, got Tensor<_, 5, 4>

    println!("// This won't compile:");
    println!("// let bad: Tensor<f32, 5, 4> = Tensor::zeros();");
    println!("// let _ = a.matmul(&bad);  // Error: 3 ≠ 5");
    println!();

    // -------------------------------------------------------------------------
    // Matrix Multiplication Values
    // -------------------------------------------------------------------------
    println!("3. Matrix Multiplication Values");
    println!("-------------------------------");

    // [1 2] × [5 6]   [19 22]
    // [3 4]   [7 8] = [43 50]
    let m1: Tensor<f32, 2, 2> = Tensor::from_data([[1.0, 2.0], [3.0, 4.0]]);
    let m2: Tensor<f32, 2, 2> = Tensor::from_data([[5.0, 6.0], [7.0, 8.0]]);
    let m3 = m1.matmul(&m2);

    println!("[1 2]   [5 6]   [19 22]");
    println!("[3 4] × [7 8] = [43 50]");
    println!();
    println!("Result: {:?}", m3.data());
    println!();

    // -------------------------------------------------------------------------
    // Element-wise Operations
    // -------------------------------------------------------------------------
    println!("4. Element-wise Operations");
    println!("--------------------------");

    let x: Tensor<f32, 1, 4> = Tensor::from_data([[-2.0, -1.0, 1.0, 2.0]]);
    let y = x.relu();

    println!("x = [-2, -1, 1, 2]");
    println!("relu(x) = {:?}", y.data());
    println!("  (negative values become 0)");
    println!();

    // -------------------------------------------------------------------------
    // Linear Layer
    // -------------------------------------------------------------------------
    println!("5. Linear Layer (Type-Safe)");
    println!("---------------------------");

    let layer: Linear<4, 2> = Linear::random();
    let input: Tensor<f32, 1, 4> = Tensor::from_data([[1.0, 2.0, 3.0, 4.0]]);
    let output: Tensor<f32, 1, 2> = layer.forward(&input);

    println!("Linear<4, 2>: (1×4) → (1×2)");
    println!("Input:  {:?}", input.data());
    println!("Output: {:?}", output.data());
    println!();

    // Batched forward pass
    let batch: Tensor<f32, 8, 4> = Tensor::zeros();
    let batch_out: Tensor<f32, 8, 2> = layer.forward(&batch);

    println!("Batched: (8×4) → (8×2)");
    println!("  Same layer, different batch size - type system tracks it!");
    let _ = batch_out; // suppress unused warning
    println!();

    // -------------------------------------------------------------------------
    // MLP (Multi-Layer Perceptron)
    // -------------------------------------------------------------------------
    println!("6. MLP (Multi-Layer Perceptron)");
    println!("-------------------------------");

    // MNIST-style architecture: 784 → 256 → 10
    let mlp: MLP<784, 256, 10> = MLP::random();
    let mnist_input: Tensor<f32, 1, 784> = Tensor::zeros();
    let mnist_output: Tensor<f32, 1, 10> = mlp.forward(&mnist_input);

    println!("MLP<784, 256, 10>: MNIST classifier");
    println!("  Input:  (1 × 784) - flattened 28×28 image");
    println!("  Hidden: (1 × 256) - after ReLU");
    println!("  Output: (1 × 10)  - class probabilities");
    println!();
    println!("Output logits: {:?}", mnist_output.data());
    println!();

    // This would NOT compile - wrong input size!
    // let bad_input: Tensor<f32, 1, 100> = Tensor::zeros();
    // let _ = mlp.forward(&bad_input);  // Error: expected 784, got 100

    println!("// This won't compile:");
    println!("// let bad: Tensor<f32, 1, 100> = Tensor::zeros();");
    println!("// mlp.forward(&bad);  // Error: expected 784, got 100");
    println!();

    // -------------------------------------------------------------------------
    // Categorical Composition
    // -------------------------------------------------------------------------
    println!("7. Categorical Composition");
    println!("--------------------------");

    // Sequential: f ; g (type system enforces cod(f) = dom(g))
    let f = |x: Tensor<f32, 1, 4>| x.add(&x); // doubles
    let g = |x: Tensor<f32, 1, 4>| x.relu();
    let fg = compose(f, g);

    let seq_input: Tensor<f32, 1, 4> = Tensor::from_data([[-1.0, 2.0, -3.0, 4.0]]);
    let seq_output = fg(seq_input);

    println!("Sequential: f ; g");
    println!("  f(x) = x + x  (double)");
    println!("  g(x) = relu(x)");
    println!("  Input:  [-1, 2, -3, 4]");
    println!("  f(x):   [-2, 4, -6, 8]");
    println!("  g(f(x)): {:?}", seq_output.data());
    println!();

    // Parallel: f ⊗ g (tensor product)
    let h1 = |x: Tensor<f32, 1, 2>| x.relu();
    let h2 = |x: Tensor<f32, 1, 3>| x.relu();
    let h1_h2 = tensor_product(h1, h2);

    let p1: Tensor<f32, 1, 2> = Tensor::from_data([[-1.0, 2.0]]);
    let p2: Tensor<f32, 1, 3> = Tensor::from_data([[-1.0, 0.0, 1.0]]);
    let (r1, r2) = h1_h2((p1, p2));

    println!("Parallel: h1 ⊗ h2");
    println!("  h1: (1×2) → (1×2)");
    println!("  h2: (1×3) → (1×3)");
    println!("  h1⊗h2: ((1×2), (1×3)) → ((1×2), (1×3))");
    println!();
    println!("  Input 1: [-1, 2]     → {:?}", r1.data());
    println!("  Input 2: [-1, 0, 1]  → {:?}", r2.data());
    println!();

    // -------------------------------------------------------------------------
    // Const vs Runtime Comparison
    // -------------------------------------------------------------------------
    println!("8. Why Const Generics Matter");
    println!("----------------------------");
    println!();
    println!("| Aspect           | Runtime Shapes      | Const Generics       |");
    println!("|------------------|---------------------|----------------------|");
    println!("| Error detection  | Runtime panic       | Compile-time error   |");
    println!("| Performance      | Bounds checks       | Zero overhead        |");
    println!("| IDE support      | Limited             | Full autocomplete    |");
    println!("| Flexibility      | Any dimensions      | Fixed at compile     |");
    println!();

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    println!("9. Summary");
    println!("----------");
    println!();
    println!("Session 7 introduced compile-time dimension checking:");
    println!();
    println!("  • Tensor<T, M, N> - dimensions in the type");
    println!("  • matmul: (M×N) × (N×P) → (M×P) - enforced by compiler");
    println!("  • Linear<IN, OUT> - layer dimensions are types");
    println!("  • MLP<IN, HIDDEN, OUT> - full network architecture as type");
    println!();
    println!("The type system IS the categorical structure:");
    println!("  • Objects = Tensor types (different dimensions = different objects)");
    println!("  • Morphisms = Functions between tensor types");
    println!("  • Composition = Type-checked function composition");
    println!();

    println!("=== Session 7 Complete ===");

    // Use c to avoid warning
    let _ = c;
}
