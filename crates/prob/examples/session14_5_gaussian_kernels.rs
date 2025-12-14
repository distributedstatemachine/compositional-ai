//! Session 14.5: Continuous Distributions and Gaussian Kernels
//!
//! Run with: cargo run -p compositional-prob --example session14_5_gaussian_kernels
//!
//! This example demonstrates:
//! - Gaussian distributions as continuous probability
//! - Gaussian kernels as morphisms in category Gauss
//! - Composition of Gaussian kernels = convolution formula
//! - The reparameterization trick for VAEs
//! - Multivariate Gaussians and their kernels
//!
//! Key insight: The same categorical structure applies to continuous distributions!

use compositional_prob::gaussian::{
    GaussianDist, GaussianKernel, MultivariateGaussian, MultivariateGaussianKernel,
};

fn main() {
    println!("=== Session 14.5: Continuous Distributions and Gaussian Kernels ===\n");

    // -------------------------------------------------------------------------
    // 1. Gaussian Distributions
    // -------------------------------------------------------------------------
    println!("1. Gaussian Distributions");
    println!("-------------------------\n");

    let standard = GaussianDist::standard();
    println!("Standard Normal N(0, 1):");
    println!("  Mean: {:.4}", standard.mean);
    println!("  Std Dev: {:.4}", standard.std_dev);
    println!("  Variance: {:.4}", standard.variance());
    println!("  pdf(0): {:.6}", standard.pdf(0.0));
    println!("  cdf(0): {:.6}", standard.cdf(0.0));
    println!("  Entropy: {:.6} nats", standard.entropy());
    println!();

    let shifted = GaussianDist::new(5.0, 2.0).unwrap();
    println!("Shifted Normal N(5, 4):");
    println!("  Mean: {:.4}", shifted.mean);
    println!("  Variance: {:.4}", shifted.variance());
    println!("  pdf(5): {:.6} (mode)", shifted.pdf(5.0));
    println!("  cdf(5): {:.6} (median)", shifted.cdf(5.0));
    println!();

    // -------------------------------------------------------------------------
    // 2. Operations on Gaussians
    // -------------------------------------------------------------------------
    println!("2. Operations on Gaussians");
    println!("--------------------------\n");

    let g1 = GaussianDist::new(2.0, 1.0).unwrap(); // N(2, 1)
    let g2 = GaussianDist::new(3.0, 2.0).unwrap(); // N(3, 4)

    println!("Given X ~ N(2, 1) and Y ~ N(3, 4):");
    println!();

    // Convolution (sum of independent Gaussians)
    let sum = g1.convolve(&g2);
    println!("X + Y ~ N({:.1}, {:.1})", sum.mean, sum.variance());
    println!("  (means add, variances add)");
    println!();

    // Scaling
    let scaled = g1.scale(3.0);
    println!("3X ~ N({:.1}, {:.1})", scaled.mean, scaled.variance());
    println!("  (mean scales by a, variance by a²)");
    println!();

    // KL divergence
    let kl = g1.kl_divergence(&g2).unwrap();
    println!("KL(N(2,1) || N(3,4)) = {:.6}", kl);
    println!();

    // -------------------------------------------------------------------------
    // 3. Gaussian Kernels: Morphisms in Category Gauss
    // -------------------------------------------------------------------------
    println!("3. Gaussian Kernels: Morphisms in Category Gauss");
    println!("-------------------------------------------------\n");

    println!("A Gaussian kernel K: x ↦ N(ax + b, σ²) is a stochastic map.");
    println!("It's a morphism in the category Gauss.\n");

    let k1 = GaussianKernel::linear(2.0, 1.0, 0.5);
    println!("K₁: x ↦ N(2x + 1, 0.25)");

    let output = k1.apply(3.0);
    println!("  K₁(3) = N({:.1}, {:.2})", output.mean, output.variance());
    println!();

    // Identity kernel
    let id = GaussianKernel::identity();
    println!("Identity: x ↦ N(x, 0) (deterministic)");
    let id_output = id.apply(5.0);
    println!(
        "  id(5) = N({:.1}, {:.1})",
        id_output.mean,
        id_output.variance()
    );
    println!();

    // Constant kernel
    let constant = GaussianKernel::constant(0.0, 1.0);
    println!("Constant: x ↦ N(0, 1) (ignores input)");
    let const_output = constant.apply(100.0);
    println!(
        "  const(100) = N({:.1}, {:.1})",
        const_output.mean,
        const_output.variance()
    );
    println!();

    // -------------------------------------------------------------------------
    // 4. Composition of Gaussian Kernels = Convolution
    // -------------------------------------------------------------------------
    println!("4. Composition of Gaussian Kernels = Convolution");
    println!("-------------------------------------------------\n");

    println!("KEY INSIGHT: When we compose Gaussian kernels:");
    println!("  K₂ ∘ K₁: x ↦ N(a₂a₁x + a₂b₁ + b₂, a₂²σ₁² + σ₂²)\n");

    let k1 = GaussianKernel::linear(2.0, 0.0, 1.0); // x ↦ N(2x, 1)
    let k2 = GaussianKernel::linear(3.0, 0.0, 2.0); // y ↦ N(3y, 4)

    println!("K₁: x ↦ N(2x, 1)  (scale by 2, add noise variance 1)");
    println!("K₂: y ↦ N(3y, 4)  (scale by 3, add noise variance 4)");
    println!();

    let composed = k2.compose(&k1);
    println!(
        "K₂ ∘ K₁: x ↦ N({}x, {})",
        composed.scale,
        composed.noise_variance()
    );
    println!();
    println!("Calculation:");
    println!("  Scale: 3 × 2 = 6");
    println!("  Noise: 3² × 1 + 4 = 9 + 4 = 13");
    println!();

    // Verify with application
    let direct = k2.apply(k1.apply(1.0).mean);
    let composed_output = composed.apply(1.0);
    println!("Verification at x=1:");
    println!("  Direct (K₂(K₁(1).mean)): N({:.1}, _)", direct.mean);
    println!(
        "  Composed ((K₂∘K₁)(1)): N({:.1}, {:.1})",
        composed_output.mean,
        composed_output.variance()
    );
    println!();

    // -------------------------------------------------------------------------
    // 5. The Reparameterization Trick
    // -------------------------------------------------------------------------
    println!("5. The Reparameterization Trick");
    println!("-------------------------------\n");

    println!("Problem: We can't backprop through z ~ N(μ, σ²)");
    println!("Solution: z = μ + σ·ε where ε ~ N(0,1)\n");

    let vae_output = GaussianDist::new(5.0, 2.0).unwrap();
    println!(
        "VAE encoder outputs: μ={:.1}, σ={:.1}",
        vae_output.mean, vae_output.std_dev
    );
    println!();

    println!("Sampling with reparameterization:");
    for epsilon in [-1.0, 0.0, 1.0, 2.0] {
        let sample = vae_output.sample_reparam(epsilon);
        println!("  ε = {:+.1} → z = {:.1}", epsilon, sample);
    }
    println!();

    println!("Kernel version: K(x) with ε gives x' = ax + b + σε");
    let k = GaussianKernel::linear(2.0, 1.0, 0.5);
    for epsilon in [-1.0, 0.0, 1.0] {
        let sample = k.sample_reparam(3.0, epsilon);
        println!("  K(3) with ε={:+.1} → {:.2}", epsilon, sample);
    }
    println!();

    // -------------------------------------------------------------------------
    // 6. Category Laws for Gauss
    // -------------------------------------------------------------------------
    println!("6. Category Laws for Gauss");
    println!("--------------------------\n");

    let k = GaussianKernel::linear(2.0, 3.0, 1.0);
    let id = GaussianKernel::identity();

    // Identity laws
    let left = id.compose(&k);
    let right = k.compose(&id);

    println!("Identity laws:");
    println!(
        "  id ∘ K = K: scale={:.1}, bias={:.1}, noise={:.1}",
        left.scale,
        left.bias,
        left.noise_variance()
    );
    println!(
        "  K ∘ id = K: scale={:.1}, bias={:.1}, noise={:.1}",
        right.scale,
        right.bias,
        right.noise_variance()
    );
    println!(
        "  K itself:   scale={:.1}, bias={:.1}, noise={:.1}",
        k.scale,
        k.bias,
        k.noise_variance()
    );
    println!();

    // Associativity
    let k1 = GaussianKernel::linear(2.0, 1.0, 0.5);
    let k2 = GaussianKernel::linear(0.5, 2.0, 1.0);
    let k3 = GaussianKernel::linear(3.0, -1.0, 0.25);

    let left_assoc = k3.compose(&k2).compose(&k1);
    let right_assoc = k3.compose(&k2.compose(&k1));

    println!("Associativity: (K₃ ∘ K₂) ∘ K₁ = K₃ ∘ (K₂ ∘ K₁)");
    println!(
        "  Left:  scale={:.2}, noise_var={:.4}",
        left_assoc.scale,
        left_assoc.noise_variance()
    );
    println!(
        "  Right: scale={:.2}, noise_var={:.4}",
        right_assoc.scale,
        right_assoc.noise_variance()
    );
    println!();

    // -------------------------------------------------------------------------
    // 7. Multivariate Gaussians
    // -------------------------------------------------------------------------
    println!("7. Multivariate Gaussians");
    println!("-------------------------\n");

    let mean = vec![1.0, 2.0];
    let cov = vec![
        1.0, 0.5, // Row 1: Var(X₁)=1, Cov(X₁,X₂)=0.5
        0.5, 2.0, // Row 2: Cov(X₂,X₁)=0.5, Var(X₂)=2
    ];
    let mv = MultivariateGaussian::new(mean, cov).unwrap();

    println!("2D Gaussian with correlation:");
    println!("  μ = [{:.1}, {:.1}]", mv.mean[0], mv.mean[1]);
    println!("  Σ = [{:.1} {:.1}]", mv.cov(0, 0), mv.cov(0, 1));
    println!("      [{:.1} {:.1}]", mv.cov(1, 0), mv.cov(1, 1));
    println!();

    println!("Marginals:");
    let m1 = mv.marginal(0);
    let m2 = mv.marginal(1);
    println!("  X₁ ~ N({:.1}, {:.1})", m1.mean, m1.variance());
    println!("  X₂ ~ N({:.1}, {:.1})", m2.mean, m2.variance());
    println!();

    // Spherical Gaussian (VAE prior)
    let prior = MultivariateGaussian::spherical(vec![0.0, 0.0, 0.0], 1.0).unwrap();
    println!("VAE prior N(0, I) in 3D:");
    println!(
        "  μ = [{:.1}, {:.1}, {:.1}]",
        prior.mean[0], prior.mean[1], prior.mean[2]
    );
    println!("  Σ = I (identity matrix)");
    println!();

    // -------------------------------------------------------------------------
    // 8. Multivariate Kernel Composition
    // -------------------------------------------------------------------------
    println!("8. Multivariate Kernel Composition");
    println!("-----------------------------------\n");

    // K₁: ℝ² → ℝ² with identity transform, unit noise
    let k1 = MultivariateGaussianKernel::new(
        vec![1.0, 0.0, 0.0, 1.0], // A = I
        vec![0.0, 0.0],           // b = 0
        vec![1.0, 0.0, 0.0, 1.0], // Σ = I
        2,
        2,
    )
    .unwrap();

    // K₂: ℝ² → ℝ² with scale=2, bias=(1,1), noise=0.5I
    let k2 = MultivariateGaussianKernel::new(
        vec![2.0, 0.0, 0.0, 2.0], // A = 2I
        vec![1.0, 1.0],           // b = (1,1)
        vec![0.5, 0.0, 0.0, 0.5], // Σ = 0.5I
        2,
        2,
    )
    .unwrap();

    println!("K₁: x ↦ N(x, I)      (identity + unit noise)");
    println!("K₂: y ↦ N(2y + 1, 0.5I)  (scale, shift, smaller noise)");
    println!();

    let composed = k2.compose(&k1).unwrap();
    println!("K₂ ∘ K₁:");
    println!("  Transform: 2I × I = 2I");
    println!("  Bias: 2I × 0 + (1,1) = (1,1)");
    println!("  Noise: 2² × I + 0.5I = 4.5I");
    println!();
    println!("  Computed noise_cov[0,0] = {:.1}", composed.noise_cov[0]);
    println!();

    // -------------------------------------------------------------------------
    // 9. Connection to Kalman Filtering
    // -------------------------------------------------------------------------
    println!("9. Connection to Kalman Filtering");
    println!("----------------------------------\n");

    println!("A Kalman filter is literally composition in Gauss!");
    println!();
    println!("State transition: x_{{t+1}} | x_t ~ N(Ax_t, Q)");
    println!("Observation:      y_t | x_t ~ N(Cx_t, R)");
    println!();

    // Simple 1D tracking example
    let process_noise: f64 = 0.1;
    let obs_noise: f64 = 1.0;

    let transition = GaussianKernel::linear(1.0, 0.0, process_noise.sqrt());
    let _observation = GaussianKernel::linear(1.0, 0.0, obs_noise.sqrt());

    println!("1D tracking:");
    println!(
        "  Transition: x_{{t+1}} | x_t ~ N(x_t, {:.1})",
        process_noise
    );
    println!("  Observation: y_t | x_t ~ N(x_t, {:.1})", obs_noise);
    println!();

    // Predict step is composition
    let predict = transition.compose(&GaussianKernel::identity());
    println!("Predict step (compose with prior):");
    println!("  Output noise variance: {:.2}", predict.noise_variance());
    println!();

    // -------------------------------------------------------------------------
    // Exercises
    // -------------------------------------------------------------------------
    println!("=== Exercises ===\n");

    println!("Exercise 1: Composition Calculation");
    println!("------------------------------------");
    println!("Given K₁: x ↦ N(2x, 1) and K₂: y ↦ N(3y, 4)");
    println!("Compute K₂ ∘ K₁.\n");

    let ex1_k1 = GaussianKernel::linear(2.0, 0.0, 1.0);
    let ex1_k2 = GaussianKernel::linear(3.0, 0.0, 2.0);
    let ex1_composed = ex1_k2.compose(&ex1_k1);

    println!("Solution:");
    println!("  K₂ ∘ K₁: x ↦ N(a₂·a₁·x, a₂²·σ₁² + σ₂²)");
    println!("         = x ↦ N(3·2·x, 3²·1 + 4)");
    println!("         = x ↦ N(6x, 13)");
    println!();
    println!(
        "  Verification: scale={}, noise_var={}",
        ex1_composed.scale,
        ex1_composed.noise_variance()
    );
    println!();

    println!("Exercise 2: Entropy of Composition");
    println!("-----------------------------------");
    println!("If H(K) = 0.5·ln(2πe·σ²) is the entropy of a kernel,");
    println!("how does H(K₂ ∘ K₁) relate to H(K₁) and H(K₂)?\n");

    let h1 = ex1_k1.entropy();
    let h2 = ex1_k2.entropy();
    let h_composed = ex1_composed.entropy();

    println!("Solution:");
    println!("  H(K₁) = {:.4} nats", h1);
    println!("  H(K₂) = {:.4} nats", h2);
    println!("  H(K₂∘K₁) = {:.4} nats", h_composed);
    println!();
    println!("  Note: H(K₂∘K₁) ≠ H(K₁) + H(K₂) because:");
    println!("  - Composed variance = a₂²σ₁² + σ₂², not σ₁² + σ₂²");
    println!("  - The scaling factor a₂ amplifies K₁'s noise");
    println!();

    println!("Exercise 3: KL Divergence Triangle");
    println!("-----------------------------------");
    println!("Compute KL(N(0,1) || N(μ,1)) for μ = 0, 1, 2, 3\n");

    let p = GaussianDist::standard();
    println!("Solution:");
    for mu in [0.0, 1.0, 2.0, 3.0] {
        let q = GaussianDist::new(mu, 1.0).unwrap();
        let kl = p.kl_divergence(&q).unwrap();
        println!("  KL(N(0,1) || N({:.0},1)) = {:.4}", mu, kl);
    }
    println!();
    println!("  Pattern: KL = μ²/2 (when variances are equal)");
    println!();

    println!("Exercise 4: Reparameterization Inverse");
    println!("--------------------------------------");
    println!("Given sample z and parameters (μ, σ), recover ε.\n");

    let mu = 5.0;
    let sigma = 2.0;
    let g = GaussianDist::new(mu, sigma).unwrap();

    println!(
        "Given N({}, {}), for each sample z, find ε such that z = μ + σε:",
        mu, sigma
    );
    for z in [3.0, 5.0, 7.0, 9.0] {
        let epsilon = g.reparam_inverse(z).unwrap();
        let reconstructed = g.sample_reparam(epsilon);
        println!(
            "  z = {:.1} → ε = {:.2} → μ + σε = {:.1}",
            z, epsilon, reconstructed
        );
    }
    println!();

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    println!("=== Session 14.5 Complete ===\n");

    println!("We implemented continuous probability via Gaussian kernels:");
    println!("  • GaussianDist: N(μ, σ²) with pdf, cdf, entropy, KL divergence");
    println!("  • GaussianKernel: x ↦ N(ax + b, σ²) as morphisms");
    println!("  • Composition: (K₂∘K₁)(x) = N(a₂a₁x + a₂b₁ + b₂, a₂²σ₁² + σ₂²)");
    println!("  • Reparameterization: z = μ + σε for backprop through sampling");
    println!("  • MultivariateGaussian: N(μ, Σ) in higher dimensions");
    println!();
    println!("Key insight: The same categorical structure applies!");
    println!("  - FinStoch: discrete, matrix multiplication");
    println!("  - Gauss: continuous, convolution formula");
    println!();
    println!("Next: Session 15 - Counterfactuals");
}
