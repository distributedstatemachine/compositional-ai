//! Session 11: Stochastic Maps (Finite Markov Kernels)
//!
//! Run with: cargo run -p compositional-prob --example session11_stochastic
//!
//! This example demonstrates:
//! - Probability distributions as vectors summing to 1
//! - Markov kernels as row-stochastic matrices
//! - Composition via matrix multiplication
//! - The category FinStoch
//! - Stationary distributions and Markov chain convergence
//!
//! Key insight: Probability is compositional. Stochastic maps form a category.

use compositional_prob::{Dist, FinStoch, Kernel, ProbError};

fn main() -> Result<(), ProbError> {
    println!("=== Session 11: Stochastic Maps (Finite Markov Kernels) ===\n");

    // -------------------------------------------------------------------------
    // 1. Probability Distributions
    // -------------------------------------------------------------------------
    println!("1. Probability Distributions");
    println!("----------------------------");
    println!();
    println!("A distribution over {{0, ..., n-1}} is a vector summing to 1.");
    println!();

    let fair_coin = Dist::uniform(2);
    println!("Fair coin:       {:?}", fair_coin.p);

    let biased_coin = Dist::new(vec![0.3, 0.7])?;
    println!("Biased coin:     {:?}", biased_coin.p);

    let fair_die = Dist::uniform(6);
    println!("Fair die:        {:?}", fair_die.p);

    let certain = Dist::point(3, 1);
    println!("Certain (idx=1): {:?}", certain.p);
    println!();

    // Entropy
    println!("Entropy (measures uncertainty):");
    println!(
        "  Fair coin:   {:.4} nats ({:.4} bits)",
        fair_coin.entropy(),
        fair_coin.entropy_bits()
    );
    println!(
        "  Biased coin: {:.4} nats ({:.4} bits)",
        biased_coin.entropy(),
        biased_coin.entropy_bits()
    );
    println!(
        "  Fair die:    {:.4} nats ({:.4} bits)",
        fair_die.entropy(),
        fair_die.entropy_bits()
    );
    println!(
        "  Certain:     {:.4} nats ({:.4} bits)",
        certain.entropy(),
        certain.entropy_bits()
    );
    println!();

    // -------------------------------------------------------------------------
    // 2. Markov Kernels (Stochastic Maps)
    // -------------------------------------------------------------------------
    println!("2. Markov Kernels (Stochastic Maps)");
    println!("-----------------------------------");
    println!();
    println!("A kernel K: n -> m is a row-stochastic matrix.");
    println!("K[i,j] = P(output=j | input=i)");
    println!();

    // Binary symmetric channel
    let channel = Kernel::new(vec![
        vec![0.9, 0.1], // P(output | input=0)
        vec![0.1, 0.9], // P(output | input=1)
    ])?;
    println!("Binary symmetric channel (error rate 0.1):");
    println!("  Row 0: {:?}", channel.k[0]);
    println!("  Row 1: {:?}", channel.k[1]);
    println!();

    // Apply to a distribution
    let message = Dist::new(vec![0.6, 0.4])?;
    let received = channel.apply(&message)?;
    println!("Send message {:?}", message.p);
    println!("Receive:     {:?}", received.p);
    println!();

    // -------------------------------------------------------------------------
    // 3. Composition of Kernels
    // -------------------------------------------------------------------------
    println!("3. Composition of Kernels");
    println!("-------------------------");
    println!();
    println!("Composing K: n->m with L: m->p gives K;L: n->p");
    println!("(K;L)[i,k] = sum_j K[i,j] * L[j,k]  (matrix multiplication)");
    println!();

    // Weather Markov chain
    let weather = Kernel::new(vec![
        vec![0.8, 0.2], // Sunny -> 80% Sunny, 20% Rainy
        vec![0.4, 0.6], // Rainy -> 40% Sunny, 60% Rainy
    ])?;
    println!("Weather transition matrix:");
    println!("  Sunny->: {:?}", weather.k[0]);
    println!("  Rainy->: {:?}", weather.k[1]);
    println!();

    // Two-day transition
    let two_days = weather.compose(&weather)?;
    println!("Two-day transition (weather ; weather):");
    println!("  Sunny->: {:?}", two_days.k[0]);
    println!("  Rainy->: {:?}", two_days.k[1]);
    println!();

    // Start sunny, forecast for several days
    let today = Dist::point(2, 0);
    println!("Starting sunny:");
    let mut forecast = today.clone();
    for day in 1..=5 {
        forecast = weather.apply(&forecast)?;
        println!(
            "  Day {}: P(sunny)={:.4}, P(rainy)={:.4}",
            day, forecast.p[0], forecast.p[1]
        );
    }
    println!();

    // -------------------------------------------------------------------------
    // 4. The Category FinStoch
    // -------------------------------------------------------------------------
    println!("4. The Category FinStoch");
    println!("------------------------");
    println!();
    println!("Objects: Natural numbers (cardinalities of finite sets)");
    println!("Morphisms: Row-stochastic matrices (Markov kernels)");
    println!("Composition: Matrix multiplication");
    println!("Identity: The identity matrix");
    println!();

    // Identity
    let id = FinStoch::id(2);
    println!("Identity on 2: {:?}", id.k);

    // Identity laws
    let k = Kernel::new(vec![vec![0.3, 0.7], vec![0.6, 0.4]])?;
    let id_then_k = FinStoch::compose(&id, &k)?;
    let k_then_id = FinStoch::compose(&k, &id)?;
    println!("id ; K = K: {}", matrix_eq(&id_then_k, &k));
    println!("K ; id = K: {}", matrix_eq(&k_then_id, &k));
    println!();

    // Associativity
    let f = Kernel::new(vec![vec![0.6, 0.4], vec![0.3, 0.7]])?;
    let g = Kernel::new(vec![vec![0.5, 0.5], vec![0.2, 0.8]])?;
    let h = Kernel::new(vec![vec![0.9, 0.1], vec![0.1, 0.9]])?;

    let fg = FinStoch::compose(&f, &g)?;
    let gh = FinStoch::compose(&g, &h)?;
    let fgh_left = FinStoch::compose(&fg, &h)?;
    let fgh_right = FinStoch::compose(&f, &gh)?;
    println!("(f;g);h = f;(g;h): {}", matrix_eq(&fgh_left, &fgh_right));
    println!();

    // -------------------------------------------------------------------------
    // 5. Distributions as Morphisms 1 -> n
    // -------------------------------------------------------------------------
    println!("5. Distributions as Morphisms 1 -> n");
    println!("------------------------------------");
    println!();
    println!("A distribution p on n elements is a morphism 1 -> n in FinStoch.");
    println!("Applying a kernel to a distribution: compose the morphisms.");
    println!();

    let p = Dist::new(vec![0.3, 0.7])?;
    let p_as_kernel = FinStoch::distribution(&p);
    println!("Distribution p = {:?}", p.p);
    println!("As kernel 1->2: {:?}", p_as_kernel.k);
    println!();

    // The discard map
    let discard = FinStoch::discard(3);
    println!("Discard map 3->1: {:?}", discard.k);
    let q = Dist::new(vec![0.2, 0.5, 0.3])?;
    let discarded = discard.apply(&q)?;
    println!("Discard({:?}) = {:?}", q.p, discarded.p);
    println!();

    // -------------------------------------------------------------------------
    // 6. Stationary Distributions
    // -------------------------------------------------------------------------
    println!("6. Stationary Distributions");
    println!("---------------------------");
    println!();
    println!("A stationary distribution satisfies: pi * K = pi");
    println!();

    let stationary = weather.stationary(1000, 1e-8).unwrap();
    println!("Weather stationary distribution:");
    println!("  pi = {:?}", stationary.p);

    // Verify it's stationary
    let after = weather.apply(&stationary)?;
    println!("  pi * K = {:?}", after.p);
    println!(
        "  Difference: {:.2e}",
        (stationary.p[0] - after.p[0]).abs() + (stationary.p[1] - after.p[1]).abs()
    );
    println!();

    // Theoretical value: solve 0.8*pi_0 + 0.4*pi_1 = pi_0 and pi_0 + pi_1 = 1
    // => pi_0 = 2/3, pi_1 = 1/3
    println!(
        "Theoretical: pi = [2/3, 1/3] = [{:.6}, {:.6}]",
        2.0 / 3.0,
        1.0 / 3.0
    );
    println!();

    // -------------------------------------------------------------------------
    // 7. Deterministic vs Stochastic
    // -------------------------------------------------------------------------
    println!("7. Deterministic vs Stochastic");
    println!("------------------------------");
    println!();

    let det = Kernel::deterministic(3, 2, |i| i % 2);
    println!("Deterministic kernel (mod 2):");
    for (i, row) in det.k.iter().enumerate() {
        println!("  {} -> {:?}", i, row);
    }
    println!("Is deterministic: {}", det.is_deterministic());
    println!();

    let stoch = Kernel::uniform(3, 2);
    println!("Uniform kernel:");
    for (i, row) in stoch.k.iter().enumerate() {
        println!("  {} -> {:?}", i, row);
    }
    println!("Is deterministic: {}", stoch.is_deterministic());
    println!();

    // -------------------------------------------------------------------------
    // 8. Information-Theoretic View
    // -------------------------------------------------------------------------
    println!("8. Information-Theoretic View");
    println!("-----------------------------");
    println!();

    let p1 = Dist::uniform(4);
    let p2 = Dist::new(vec![0.7, 0.1, 0.1, 0.1])?;

    println!("p1 (uniform): {:?}", p1.p);
    println!("p2 (peaked):  {:?}", p2.p);
    println!();

    println!("Entropy:");
    println!("  H(p1) = {:.4} bits", p1.entropy_bits());
    println!("  H(p2) = {:.4} bits", p2.entropy_bits());
    println!();

    println!("Total variation distance:");
    let tv = p1.tv_distance(&p2)?;
    println!("  TV(p1, p2) = {:.4}", tv);
    println!();

    println!("KL divergence:");
    let kl_12 = p1.kl_divergence(&p2)?;
    let kl_21 = p2.kl_divergence(&p1)?;
    println!("  KL(p1 || p2) = {:.4}", kl_12);
    println!("  KL(p2 || p1) = {:.4}", kl_21);
    println!("  (Note: KL is asymmetric)");
    println!();

    // -------------------------------------------------------------------------
    // 9. Tensor Products
    // -------------------------------------------------------------------------
    println!("9. Tensor Products");
    println!("------------------");
    println!();
    println!("K tensor L represents independent parallel processes.");
    println!();

    let flip1 = Kernel::new(vec![vec![0.5, 0.5]])?; // Fair coin
    let flip2 = Kernel::new(vec![vec![0.3, 0.7]])?; // Biased coin

    let two_coins = flip1.tensor(&flip2);
    println!("Fair coin tensor Biased coin:");
    println!("  Joint outcomes: (0,0), (0,1), (1,0), (1,1)");
    println!("  Probabilities: {:?}", two_coins.k[0]);
    println!("  (0.5*0.3, 0.5*0.7, 0.5*0.3, 0.5*0.7) = (0.15, 0.35, 0.15, 0.35)");
    println!();

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    println!("=== Session 11 Complete ===");
    println!();
    println!("We implemented the category of finite stochastic maps:");
    println!("  - Distributions: vectors summing to 1");
    println!("  - Kernels: row-stochastic matrices");
    println!("  - Composition: matrix multiplication");
    println!("  - Category laws: identity and associativity");
    println!("  - Stationary distributions via power iteration");
    println!();
    println!("Key insight: Probability theory has compositional structure.");
    println!("Stochastic maps form a category where morphisms are");
    println!("conditional distributions and composition is marginalization.");
    println!();
    println!("Next: Session 12 - Bayesian Networks as Composed Kernels");

    Ok(())
}

fn matrix_eq(a: &Kernel, b: &Kernel) -> bool {
    if a.n_inputs != b.n_inputs || a.n_outputs != b.n_outputs {
        return false;
    }
    for i in 0..a.n_inputs {
        for j in 0..a.n_outputs {
            if (a.k[i][j] - b.k[i][j]).abs() > 1e-6 {
                return false;
            }
        }
    }
    true
}
