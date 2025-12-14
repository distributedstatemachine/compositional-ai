//! Session 12: Bayesian Networks as Composed Kernels
//!
//! Run with: cargo run -p compositional-prob --example session12_bayesnets
//!
//! This example demonstrates:
//! - Bayesian networks as compositions of Markov kernels
//! - Joint distribution factorization
//! - Marginalization (summing out variables)
//! - Conditional queries (inference)
//! - The classic "sprinkler" network
//!
//! Key insight: A Bayes net is a string diagram in FinStoch.

use compositional_prob::{sprinkler_network, BayesNet, Dist, Kernel};
use std::collections::HashMap;

fn main() {
    println!("=== Session 12: Bayesian Networks as Composed Kernels ===\n");

    // -------------------------------------------------------------------------
    // 1. Bayesian Networks = Composed Conditionals
    // -------------------------------------------------------------------------
    println!("1. Bayesian Networks = Composed Conditionals");
    println!("--------------------------------------------");
    println!();
    println!("A Bayesian network factors a joint distribution:");
    println!("  P(X₁, ..., Xₙ) = ∏ᵢ P(Xᵢ | parents(Xᵢ))");
    println!();
    println!("Each conditional P(Xᵢ | parents) is a Markov kernel!");
    println!();

    // -------------------------------------------------------------------------
    // 2. The Classic Sprinkler Network
    // -------------------------------------------------------------------------
    println!("2. The Classic Sprinkler Network");
    println!("--------------------------------");
    println!();
    println!("Structure:");
    println!("       Cloudy (C)");
    println!("       ↙     ↘");
    println!("  Sprinkler   Rain");
    println!("      (S)      (R)");
    println!("         ↘    ↙");
    println!("        WetGrass (W)");
    println!();

    let net = sprinkler_network();
    println!("Variables: {:?}", net.var_names);
    println!("States per variable: {:?}", net.var_states);
    println!("Number of factors: {}", net.factors.len());
    println!();

    // -------------------------------------------------------------------------
    // 3. Joint Distribution
    // -------------------------------------------------------------------------
    println!("3. Joint Distribution");
    println!("---------------------");
    println!();
    println!("P(C,S,R,W) = P(C) · P(S|C) · P(R|C) · P(W|S,R)");
    println!();

    let joint = net.full_joint();
    println!("Total joint states: {} (2⁴ = 16)", joint.probs.len());
    println!();

    // Show some joint probabilities
    println!("Sample joint probabilities:");
    let assignments = vec![
        vec![0, 0, 0, 0], // Not cloudy, no sprinkler, no rain, dry
        vec![1, 0, 1, 1], // Cloudy, no sprinkler, rain, wet
        vec![0, 1, 0, 1], // Not cloudy, sprinkler, no rain, wet
        vec![1, 1, 1, 1], // Cloudy, sprinkler, rain, wet
    ];

    for assignment in &assignments {
        let prob = net.joint_prob(assignment);
        println!(
            "  P(C={}, S={}, R={}, W={}) = {:.6}",
            assignment[0], assignment[1], assignment[2], assignment[3], prob
        );
    }
    println!();

    // Verify sums to 1
    let sum: f32 = joint.probs.iter().sum();
    println!("Sum of all joint probabilities: {:.6}", sum);
    println!();

    // -------------------------------------------------------------------------
    // 4. Marginalization
    // -------------------------------------------------------------------------
    println!("4. Marginalization");
    println!("------------------");
    println!();
    println!("Marginalization = summing out variables");
    println!("P(X) = Σ_{{other vars}} P(X, other vars)");
    println!();

    // Marginal distributions
    let p_cloudy = net.marginal(0).unwrap();
    let p_sprinkler = net.marginal(1).unwrap();
    let p_rain = net.marginal(2).unwrap();
    let p_wet = net.marginal(3).unwrap();

    println!("Marginal distributions:");
    println!(
        "  P(Cloudy)    = [False: {:.4}, True: {:.4}]",
        p_cloudy.p[0], p_cloudy.p[1]
    );
    println!(
        "  P(Sprinkler) = [False: {:.4}, True: {:.4}]",
        p_sprinkler.p[0], p_sprinkler.p[1]
    );
    println!(
        "  P(Rain)      = [False: {:.4}, True: {:.4}]",
        p_rain.p[0], p_rain.p[1]
    );
    println!(
        "  P(WetGrass)  = [False: {:.4}, True: {:.4}]",
        p_wet.p[0], p_wet.p[1]
    );
    println!();

    // -------------------------------------------------------------------------
    // 5. Conditional Queries (Inference)
    // -------------------------------------------------------------------------
    println!("5. Conditional Queries (Inference)");
    println!("----------------------------------");
    println!();
    println!("Given evidence, compute posterior: P(query | evidence)");
    println!();

    // P(Rain | Cloudy=True)
    let mut evidence = HashMap::new();
    evidence.insert(0, 1); // Cloudy = True
    let p_rain_given_cloudy = net.query(2, &evidence).unwrap();
    println!("P(Rain | Cloudy=True):");
    println!(
        "  [False: {:.4}, True: {:.4}]",
        p_rain_given_cloudy.p[0], p_rain_given_cloudy.p[1]
    );
    println!("  (Should be close to [0.2, 0.8] from the CPT)");
    println!();

    // P(Cloudy | WetGrass=True) - "explaining away"
    let mut evidence2 = HashMap::new();
    evidence2.insert(3, 1); // WetGrass = True
    let p_cloudy_given_wet = net.query(0, &evidence2).unwrap();
    println!("P(Cloudy | WetGrass=True):");
    println!(
        "  [False: {:.4}, True: {:.4}]",
        p_cloudy_given_wet.p[0], p_cloudy_given_wet.p[1]
    );
    println!();

    // P(Sprinkler | WetGrass=True) - more explaining away
    let p_sprinkler_given_wet = net.query(1, &evidence2).unwrap();
    println!("P(Sprinkler | WetGrass=True):");
    println!(
        "  [False: {:.4}, True: {:.4}]",
        p_sprinkler_given_wet.p[0], p_sprinkler_given_wet.p[1]
    );
    println!();

    // P(Sprinkler | WetGrass=True, Rain=True) - conditional independence broken
    let mut evidence3 = HashMap::new();
    evidence3.insert(3, 1); // WetGrass = True
    evidence3.insert(2, 1); // Rain = True
    let p_sprinkler_given_wet_rain = net.query(1, &evidence3).unwrap();
    println!("P(Sprinkler | WetGrass=True, Rain=True):");
    println!(
        "  [False: {:.4}, True: {:.4}]",
        p_sprinkler_given_wet_rain.p[0], p_sprinkler_given_wet_rain.p[1]
    );
    println!("  (Rain 'explains' the wet grass, so sprinkler less likely)");
    println!();

    // -------------------------------------------------------------------------
    // 6. Building Networks from Kernels
    // -------------------------------------------------------------------------
    println!("6. Building Networks from Kernels");
    println!("---------------------------------");
    println!();
    println!("Each CPT is a Markov kernel P(child | parents):");
    println!();

    // Show the kernels
    println!("P(Cloudy) - prior:");
    println!("  [0.5, 0.5]");
    println!();

    println!("P(Sprinkler | Cloudy) - kernel 2→2:");
    println!("  Cloudy=F: [0.5, 0.5]  (50% sprinkler when clear)");
    println!("  Cloudy=T: [0.9, 0.1]  (10% sprinkler when cloudy)");
    println!();

    println!("P(Rain | Cloudy) - kernel 2→2:");
    println!("  Cloudy=F: [0.8, 0.2]  (20% rain when clear)");
    println!("  Cloudy=T: [0.2, 0.8]  (80% rain when cloudy)");
    println!();

    println!("P(WetGrass | Sprinkler, Rain) - kernel 4→2:");
    println!("  (S=F,R=F): [1.0, 0.0]   (dry)");
    println!("  (S=T,R=F): [0.1, 0.9]   (sprinkler → wet)");
    println!("  (S=F,R=T): [0.2, 0.8]   (rain → wet)");
    println!("  (S=T,R=T): [0.01, 0.99] (both → very wet)");
    println!();

    // -------------------------------------------------------------------------
    // 7. Custom Network: Simple Chain
    // -------------------------------------------------------------------------
    println!("7. Custom Network: Simple Chain");
    println!("-------------------------------");
    println!();
    println!("Building: A → B → C");
    println!();

    let chain = BayesNet::new(vec![2, 2, 2]);
    let mut chain = chain.with_names(vec!["A", "B", "C"]);

    // P(A) = [0.3, 0.7]
    chain
        .add_prior(0, &Dist::new(vec![0.3, 0.7]).unwrap())
        .unwrap();

    // P(B|A)
    chain
        .add_conditional(
            1,
            vec![0],
            Kernel::new(vec![vec![0.9, 0.1], vec![0.4, 0.6]]).unwrap(),
        )
        .unwrap();

    // P(C|B)
    chain
        .add_conditional(
            2,
            vec![1],
            Kernel::new(vec![vec![0.8, 0.2], vec![0.3, 0.7]]).unwrap(),
        )
        .unwrap();

    println!("Chain network built with {} factors", chain.factors.len());

    let p_a = chain.marginal(0).unwrap();
    let p_b = chain.marginal(1).unwrap();
    let p_c = chain.marginal(2).unwrap();

    println!("Marginals:");
    println!("  P(A) = [{:.4}, {:.4}]", p_a.p[0], p_a.p[1]);
    println!("  P(B) = [{:.4}, {:.4}]", p_b.p[0], p_b.p[1]);
    println!("  P(C) = [{:.4}, {:.4}]", p_c.p[0], p_c.p[1]);
    println!();

    // -------------------------------------------------------------------------
    // 8. The Categorical Perspective
    // -------------------------------------------------------------------------
    println!("8. The Categorical Perspective");
    println!("------------------------------");
    println!();
    println!("A Bayesian network is a string diagram in FinStoch:");
    println!();
    println!("  • Each node = random variable (object)");
    println!("  • Each CPT = Markov kernel (morphism)");
    println!("  • DAG structure = wiring of the diagram");
    println!("  • Joint distribution = composition of morphisms");
    println!();
    println!("Marginalization = composition with discard:");
    println!("  discard: X → 1 (the unique map to terminal object)");
    println!();
    println!("Conditioning = restriction + renormalization:");
    println!("  P(A|B=b) ∝ P(A,B=b)");
    println!();

    // -------------------------------------------------------------------------
    // 9. Explaining Away
    // -------------------------------------------------------------------------
    println!("9. Explaining Away");
    println!("------------------");
    println!();
    println!("Classic example from the sprinkler network:");
    println!();
    println!(
        "Prior P(Sprinkler): [{:.4}, {:.4}]",
        p_sprinkler.p[0], p_sprinkler.p[1]
    );
    println!();
    println!("Observing wet grass increases belief in sprinkler:");
    println!(
        "  P(Sprinkler | Wet=T): [{:.4}, {:.4}]",
        p_sprinkler_given_wet.p[0], p_sprinkler_given_wet.p[1]
    );
    println!();
    println!("But knowing it rained 'explains away' the wetness:");
    println!(
        "  P(Sprinkler | Wet=T, Rain=T): [{:.4}, {:.4}]",
        p_sprinkler_given_wet_rain.p[0], p_sprinkler_given_wet_rain.p[1]
    );
    println!();
    println!("Rain and Sprinkler become negatively correlated given WetGrass!");
    println!("This is the 'explaining away' effect (v-structure).");
    println!();

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    println!("=== Session 12 Complete ===");
    println!();
    println!("We implemented Bayesian networks as composed kernels:");
    println!("  • BayesNet: collection of factors over a DAG");
    println!("  • Factor: CPT as Markov kernel P(child|parents)");
    println!("  • Joint distribution from factorization");
    println!("  • Marginalization by summing out variables");
    println!("  • Conditioning via Bayes' rule");
    println!();
    println!("Key insight: Bayes nets are string diagrams in FinStoch.");
    println!("The DAG tells us how to compose the conditional kernels.");
    println!();
    println!("Next: Session 13 - Hidden Markov Models");
}
