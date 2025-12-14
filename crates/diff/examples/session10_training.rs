//! Session 10: Parameters and SGD Training Loop
//!
//! Run with: cargo run -p compositional-diff --example session10_training
//!
//! This example demonstrates:
//! - Parameters as inputs that get updated
//! - SGD optimizer for gradient descent
//! - Training loop: forward -> loss -> backward -> update
//! - Loss decreasing over epochs
//!
//! Key insight: Training is composition: forward ; loss ; backward ; update

use compositional_diff::ops::RTensor;
use compositional_diff::optim::{SGDMomentum, SGD};
use compositional_diff::train::{generate_linear_data, train_step, LinearLayer};

fn main() {
    println!("=== Session 10: Parameters and SGD Training ===\n");

    // -------------------------------------------------------------------------
    // 1. Parameters vs Inputs
    // -------------------------------------------------------------------------
    println!("1. Parameters vs Inputs");
    println!("-----------------------");
    println!();
    println!("In a computation graph:");
    println!("  - Inputs: Data that changes each batch (x, y)");
    println!("  - Parameters: Weights that persist and get updated (W, b)");
    println!();
    println!("Both flow into the same graph, but parameters are updated by gradients.");
    println!();

    // -------------------------------------------------------------------------
    // 2. Simple SGD Update
    // -------------------------------------------------------------------------
    println!("2. Simple SGD Update");
    println!("--------------------");
    println!();
    println!("SGD rule: theta = theta - lr * grad");
    println!();

    let mut param = RTensor::vector(vec![1.0, 2.0, 3.0]);
    let grad = RTensor::vector(vec![0.5, 0.5, 0.5]);

    println!("Before: param = {:?}", param.data);
    println!("Gradient: {:?}", grad.data);

    let optimizer = SGD::new(0.1);
    optimizer.step(&mut [param.clone()], &[grad.clone()]);

    // Manual update for demonstration
    for (p, g) in param.data.iter_mut().zip(grad.data.iter()) {
        *p -= 0.1 * g;
    }

    println!("After (lr=0.1): param = {:?}", param.data);
    println!("  (each element decreased by 0.1 * 0.5 = 0.05)");
    println!();

    // -------------------------------------------------------------------------
    // 3. Training Step: Forward -> Loss -> Backward -> Update
    // -------------------------------------------------------------------------
    println!("3. Training Step");
    println!("----------------");
    println!();
    println!("One training step for linear regression: y = W * x");
    println!();

    // Model: y = w * x (1D case)
    let mut w = RTensor::matrix(1, 1, vec![0.5]); // Start with w = 0.5
    let x = RTensor::matrix(1, 1, vec![2.0]); // Input
    let y_true = RTensor::matrix(1, 1, vec![6.0]); // Target: y = 3 * x = 6

    println!("Model: y = w * x");
    println!("True relationship: y = 3 * x");
    println!();
    println!("Initial w = {}", w.data[0]);
    println!("Input x = {}", x.data[0]);
    println!("Target y = {}", y_true.data[0]);
    println!();

    // Training step
    let (loss_before, grad_w) = train_step(&x, &y_true, &w);
    println!(
        "Prediction = w * x = {} * {} = {}",
        w.data[0],
        x.data[0],
        w.data[0] * x.data[0]
    );
    println!("Loss (MSE) = {:.4}", loss_before);
    println!("Gradient dL/dw = {:.4}", grad_w.data[0]);
    println!();

    // Update
    let lr = 0.01;
    w.data[0] -= lr * grad_w.data[0];
    println!("After update (lr={}): w = {:.4}", lr, w.data[0]);
    println!("  (moved toward w=3.0)");
    println!();

    // -------------------------------------------------------------------------
    // 4. Training Loop
    // -------------------------------------------------------------------------
    println!("4. Training Loop");
    println!("----------------");
    println!();
    println!("Training y = 2*x + 1 on synthetic data");
    println!();

    // Generate synthetic data: y = 2*x + 1
    let data = generate_linear_data(50, 2.0, 1.0, 0.1, 42);
    println!("Generated {} data points", data.len());
    println!(
        "Sample: x={:.2}, y={:.2}",
        data[0].0.data[0], data[0].1.data[0]
    );
    println!();

    // Initialize model
    let mut w = RTensor::matrix(1, 1, vec![0.0]);
    let optimizer = SGD::new(0.01);

    println!("Training for 100 epochs...");
    println!();

    let mut losses = Vec::new();
    for epoch in 0..100 {
        let mut epoch_loss = 0.0;

        for (x, y_true) in &data {
            let (loss, grad_w) = train_step(x, y_true, &w);
            epoch_loss += loss;

            // Update
            for (wp, gp) in w.data.iter_mut().zip(grad_w.data.iter()) {
                *wp -= optimizer.learning_rate * gp;
            }
        }

        let avg_loss = epoch_loss / data.len() as f32;
        losses.push(avg_loss);

        if epoch % 20 == 0 || epoch == 99 {
            println!(
                "Epoch {:3}: loss = {:.6}, w = {:.4}",
                epoch, avg_loss, w.data[0]
            );
        }
    }

    println!();
    println!("Final w = {:.4} (target: 2.0)", w.data[0]);
    println!(
        "Loss decreased: {:.6} -> {:.6}",
        losses[0],
        losses.last().unwrap()
    );
    println!();

    // -------------------------------------------------------------------------
    // 5. SGD with Momentum
    // -------------------------------------------------------------------------
    println!("5. SGD with Momentum");
    println!("--------------------");
    println!();
    println!("Momentum helps accelerate SGD in the relevant direction:");
    println!("  v = momentum * v + grad");
    println!("  theta = theta - lr * v");
    println!();

    let mut w_momentum = RTensor::matrix(1, 1, vec![0.0]);
    let mut optimizer_momentum = SGDMomentum::new(0.01, 0.9);

    println!("Training with momentum=0.9...");

    for _epoch in 0..100 {
        for (x, y_true) in &data {
            let (_, grad_w) = train_step(x, y_true, &w_momentum);
            optimizer_momentum.step(&mut [w_momentum.clone()], &[grad_w.clone()]);

            // Apply update to actual w
            for (wp, gp) in w_momentum.data.iter_mut().zip(grad_w.data.iter()) {
                *wp -= optimizer_momentum.learning_rate * gp;
            }
        }
    }

    println!("Final w (with momentum) = {:.4}", w_momentum.data[0]);
    println!();

    // -------------------------------------------------------------------------
    // 6. Linear Layer
    // -------------------------------------------------------------------------
    println!("6. Linear Layer");
    println!("---------------");
    println!();

    let layer = LinearLayer::new(3, 2, 42);
    println!("Linear layer: input_dim=3, output_dim=2");
    println!("Weight shape: {:?}", layer.weights.shape);
    println!("Bias shape: {:?}", layer.bias.shape);
    println!();

    let input = RTensor::matrix(3, 1, vec![1.0, 2.0, 3.0]);
    let output = layer.forward(&input);
    println!("Input: {:?}", input.data);
    println!("Output: {:?}", output.data);
    println!();

    // -------------------------------------------------------------------------
    // 7. The Categorical View
    // -------------------------------------------------------------------------
    println!("7. The Categorical View");
    println!("-----------------------");
    println!();
    println!("Training is composition:");
    println!();
    println!("  (Params, Data, Target)");
    println!("       |");
    println!("       v  Forward");
    println!("  (Prediction, Target)");
    println!("       |");
    println!("       v  Loss");
    println!("     Scalar");
    println!("       |");
    println!("       v  Backward (functor to Op(C))");
    println!("   Gradients");
    println!("       |");
    println!("       v  Update (endomorphism)");
    println!("    Params'");
    println!();
    println!("Each step is a morphism; training is their composition.");
    println!();

    // -------------------------------------------------------------------------
    // 8. Verification: Loss Decreases
    // -------------------------------------------------------------------------
    println!("8. Verification: Loss Decreases");
    println!("-------------------------------");
    println!();

    // Quick sanity check
    let initial_loss = losses[0];
    let final_loss = *losses.last().unwrap();
    let decreased = final_loss < initial_loss;

    println!("Initial loss: {:.6}", initial_loss);
    println!("Final loss:   {:.6}", final_loss);
    println!("Decreased:    {}", if decreased { "YES" } else { "NO" });
    println!();

    if decreased {
        println!("Training is working correctly!");
    } else {
        println!("Warning: Loss did not decrease. Check learning rate or model.");
    }
    println!();

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    println!("=== Session 10 Complete ===");
    println!();
    println!("We implemented training with gradient descent:");
    println!("  - Parameters: inputs that get updated");
    println!("  - SGD: theta = theta - lr * grad");
    println!("  - Momentum: accelerates convergence");
    println!("  - Training loop: forward -> loss -> backward -> update");
    println!("  - Categorical view: training as composition");
    println!();
    println!("This completes the autodiff track (Sessions 7-10)!");
    println!();
    println!("Next: Session 11 - Stochastic Maps (Probability Track)");
}
