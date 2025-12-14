//! Session 13: Conditioning and Bayes' Rule
//!
//! Run with: cargo run -p compositional-prob --example session13_conditioning
//!
//! This example demonstrates:
//! - Why conditioning requires renormalization (vs composition)
//! - Bayes' rule: P(H|E) = P(E|H)·P(H) / P(E)
//! - Sequential Bayesian updates
//! - The base rate fallacy (medical diagnosis)
//! - Likelihood ratios and odds
//! - Importance sampling and likelihood weighting
//!
//! Key insight: Conditioning is NOT plain composition!

use compositional_prob::{
    bayes_update, evidence_probability, importance_sample, likelihood_ratio,
    likelihood_weighting_with_diagnostics, mutual_information, odds_to_prob, odds_update,
    prob_to_odds, sequential_update, sprinkler_network, Dist, ExactInference, Kernel,
};
use std::collections::HashMap;

fn main() {
    println!("=== Session 13: Conditioning and Bayes' Rule ===\n");

    // -------------------------------------------------------------------------
    // 1. Why Conditioning Requires Renormalization
    // -------------------------------------------------------------------------
    println!("1. Why Conditioning Requires Renormalization");
    println!("--------------------------------------------");
    println!();
    println!("Composition preserves normalization automatically:");
    println!("  Q(y) = Σₓ P(x) · K(y|x)  → always sums to 1");
    println!();
    println!("But conditioning restricts the probability space:");
    println!("  P(X|E=e) ∝ P(X) · P(E=e|X)  → needs renormalization!");
    println!();

    // Demonstrate with a simple example
    let prior = Dist::new(vec![0.6, 0.4]).unwrap();
    let likelihood = Kernel::new(vec![
        vec![0.8, 0.2], // H=0: P(E|H=0)
        vec![0.3, 0.7], // H=1: P(E|H=1)
    ])
    .unwrap();

    println!("Prior P(H):      [{:.2}, {:.2}]", prior.p[0], prior.p[1]);
    println!(
        "Likelihood P(E|H=0): [{:.2}, {:.2}]",
        likelihood.k[0][0], likelihood.k[0][1]
    );
    println!(
        "Likelihood P(E|H=1): [{:.2}, {:.2}]",
        likelihood.k[1][0], likelihood.k[1][1]
    );
    println!();

    // Observe E=1
    let unnorm_h0 = prior.p[0] * likelihood.k[0][1]; // P(H=0) · P(E=1|H=0)
    let unnorm_h1 = prior.p[1] * likelihood.k[1][1]; // P(H=1) · P(E=1|H=1)
    println!("Unnormalized posterior (E=1):");
    println!(
        "  P(H=0) · P(E=1|H=0) = {:.2} × {:.2} = {:.4}",
        prior.p[0], likelihood.k[0][1], unnorm_h0
    );
    println!(
        "  P(H=1) · P(E=1|H=1) = {:.2} × {:.2} = {:.4}",
        prior.p[1], likelihood.k[1][1], unnorm_h1
    );
    println!("  Sum = {:.4} ≠ 1.0", unnorm_h0 + unnorm_h1);
    println!();

    let posterior = bayes_update(&prior, &likelihood, 1);
    println!("After normalization (Bayes' rule):");
    println!(
        "  P(H|E=1) = [{:.4}, {:.4}]",
        posterior.p[0], posterior.p[1]
    );
    println!("  Sum = {:.4}", posterior.p[0] + posterior.p[1]);
    println!();

    // -------------------------------------------------------------------------
    // 2. Bayes' Rule Formula
    // -------------------------------------------------------------------------
    println!("2. Bayes' Rule Formula");
    println!("----------------------");
    println!();
    println!("P(H|E) = P(E|H) · P(H) / P(E)");
    println!();
    println!("Where:");
    println!("  P(H)   = Prior (belief before evidence)");
    println!("  P(E|H) = Likelihood (evidence probability given hypothesis)");
    println!("  P(H|E) = Posterior (belief after evidence)");
    println!("  P(E)   = Evidence (normalizing constant)");
    println!();

    let p_e = evidence_probability(&prior, &likelihood, 1);
    println!("In our example:");
    println!("  P(E=1) = P(H=0)·P(E=1|H=0) + P(H=1)·P(E=1|H=1)");
    println!(
        "         = {:.2}×{:.2} + {:.2}×{:.2} = {:.4}",
        prior.p[0], likelihood.k[0][1], prior.p[1], likelihood.k[1][1], p_e
    );
    println!();
    println!("  P(H=1|E=1) = P(E=1|H=1)·P(H=1) / P(E=1)");
    println!(
        "            = {:.2}×{:.2} / {:.4} = {:.4}",
        likelihood.k[1][1], prior.p[1], p_e, posterior.p[1]
    );
    println!();

    // -------------------------------------------------------------------------
    // 3. Medical Diagnosis: The Base Rate Fallacy
    // -------------------------------------------------------------------------
    println!("3. Medical Diagnosis: The Base Rate Fallacy");
    println!("-------------------------------------------");
    println!();
    println!("A classic example of why Bayes' rule matters:");
    println!();

    // Disease prevalence: 1%
    let disease_prior = Dist::new(vec![0.99, 0.01]).unwrap();

    // Test accuracy
    let test = Kernel::new(vec![
        vec![0.95, 0.05], // Healthy: 5% false positive
        vec![0.10, 0.90], // Diseased: 90% true positive (sensitivity)
    ])
    .unwrap();

    println!("Disease prevalence: 1%");
    println!("Test sensitivity (true positive rate): 90%");
    println!("Test specificity (true negative rate): 95%");
    println!();

    // Patient tests positive
    let posterior_disease = bayes_update(&disease_prior, &test, 1);
    let p_positive = evidence_probability(&disease_prior, &test, 1);

    println!("A patient tests positive. What's P(Disease | Positive)?");
    println!();
    println!("Naive intuition: 'The test is 90% accurate, so ~90% chance'");
    println!();
    println!("Bayes' rule calculation:");
    println!("  P(+) = P(Healthy)·P(+|Healthy) + P(Disease)·P(+|Disease)");
    println!("       = 0.99×0.05 + 0.01×0.90 = {:.4}", p_positive);
    println!();
    println!("  P(Disease|+) = P(+|Disease)·P(Disease) / P(+)");
    println!(
        "               = 0.90×0.01 / {:.4} = {:.4}",
        p_positive, posterior_disease.p[1]
    );
    println!();
    println!(
        "Actual probability: {:.1}%!",
        posterior_disease.p[1] * 100.0
    );
    println!();
    println!("The base rate (1% prevalence) dominates!");
    println!("Most positive tests are false positives from the healthy majority.");
    println!();

    // -------------------------------------------------------------------------
    // 4. Sequential Updates
    // -------------------------------------------------------------------------
    println!("4. Sequential Updates");
    println!("---------------------");
    println!();
    println!("Multiple evidence updates in sequence:");
    println!("P(H|E₁,E₂) ∝ P(E₂|H) · P(H|E₁)");
    println!();

    // Two independent tests
    let test2 = Kernel::new(vec![
        vec![0.92, 0.08], // Different test: 8% false positive
        vec![0.15, 0.85], // 85% true positive
    ])
    .unwrap();

    println!("Patient takes two independent tests...");
    println!();

    // Sequential updates
    let after_test1 = bayes_update(&disease_prior, &test, 1);
    println!("After Test 1 (positive):");
    println!("  P(Disease) = {:.4}", after_test1.p[1]);

    let after_test2 = bayes_update(&after_test1, &test2, 1);
    println!("After Test 2 (also positive):");
    println!("  P(Disease) = {:.4}", after_test2.p[1]);

    // Or all at once
    let observations = vec![(&test, 1_usize), (&test2, 1_usize)];
    let combined = sequential_update(&disease_prior, &observations);
    println!();
    println!("Combined (should match): P(Disease) = {:.4}", combined.p[1]);
    println!();
    println!("Two positive tests significantly increase disease probability!");
    println!();

    // -------------------------------------------------------------------------
    // 5. Likelihood Ratios and Odds
    // -------------------------------------------------------------------------
    println!("5. Likelihood Ratios and Odds");
    println!("-----------------------------");
    println!();
    println!("Alternative formulation using odds:");
    println!("  posterior_odds = prior_odds × likelihood_ratio");
    println!();

    let prior_odds = prob_to_odds(0.01); // 1% disease
    let lr = likelihood_ratio(&test, 1); // LR for positive test

    println!("Prior odds: {:.4} (1:99 against disease)", prior_odds);
    println!("Likelihood ratio for positive test: {:.2}", lr);
    println!(
        "  = P(+|Disease) / P(+|Healthy) = 0.90 / 0.05 = {:.2}",
        0.90 / 0.05
    );
    println!();

    let posterior_odds = odds_update(prior_odds, lr);
    let posterior_prob = odds_to_prob(posterior_odds);

    println!("Posterior odds: {:.4}", posterior_odds);
    println!("Posterior probability: {:.4}", posterior_prob);
    println!();
    println!(
        "This matches our earlier calculation: {:.4}",
        posterior_disease.p[1]
    );
    println!();

    // -------------------------------------------------------------------------
    // 6. Inference in Bayesian Networks
    // -------------------------------------------------------------------------
    println!("6. Inference in Bayesian Networks");
    println!("---------------------------------");
    println!();

    let net = sprinkler_network();
    let mut engine = ExactInference::new(&net);

    println!("Using the sprinkler network:");
    println!("  Cloudy → Sprinkler, Rain → WetGrass");
    println!();

    // Marginals
    let p_wet = engine.marginal(3).unwrap();
    println!("Marginal P(WetGrass):");
    println!("  [Dry: {:.4}, Wet: {:.4}]", p_wet.p[0], p_wet.p[1]);
    println!();

    // Conditional queries
    let mut evidence = HashMap::new();
    evidence.insert(3, 1); // WetGrass = True

    let p_rain_given_wet = engine.query(2, &evidence).unwrap();
    println!("P(Rain | WetGrass=True):");
    println!(
        "  [No: {:.4}, Yes: {:.4}]",
        p_rain_given_wet.p[0], p_rain_given_wet.p[1]
    );

    let p_sprinkler_given_wet = engine.query(1, &evidence).unwrap();
    println!("P(Sprinkler | WetGrass=True):");
    println!(
        "  [Off: {:.4}, On: {:.4}]",
        p_sprinkler_given_wet.p[0], p_sprinkler_given_wet.p[1]
    );
    println!();

    // Evidence probability
    let p_evidence = engine.evidence_prob(&evidence);
    println!("P(WetGrass=True) = {:.4}", p_evidence);
    println!();

    // -------------------------------------------------------------------------
    // 7. Information Theory
    // -------------------------------------------------------------------------
    println!("7. Information Theory");
    println!("---------------------");
    println!();
    println!("Mutual information measures dependence between variables:");
    println!();

    let mi_cloudy_rain = mutual_information(&net, 0, 2);
    let mi_sprinkler_rain = mutual_information(&net, 1, 2);
    let mi_cloudy_wet = mutual_information(&net, 0, 3);

    println!("I(Cloudy; Rain) = {:.4} nats", mi_cloudy_rain);
    println!("  (Direct causal link)");
    println!();
    println!("I(Sprinkler; Rain) = {:.4} nats", mi_sprinkler_rain);
    println!("  (Common cause: Cloudy)");
    println!();
    println!("I(Cloudy; WetGrass) = {:.4} nats", mi_cloudy_wet);
    println!("  (Indirect: through Rain and Sprinkler)");
    println!();

    // -------------------------------------------------------------------------
    // 8. Importance Sampling
    // -------------------------------------------------------------------------
    println!("8. Importance Sampling");
    println!("----------------------");
    println!();
    println!("When exact inference is intractable, we can sample!");
    println!();

    // Basic importance sampling
    let target = Dist::new(vec![0.1, 0.2, 0.3, 0.4]).unwrap();
    let proposal = Dist::uniform(4);

    println!("Target distribution:  [0.1, 0.2, 0.3, 0.4]");
    println!("Proposal distribution: uniform");
    println!();

    let f = |x: usize| x as f32;
    let result = importance_sample(&target, &proposal, f, 10000, Some(42));

    println!("Estimating E[X] (true value = 2.0):");
    println!("  Estimate: {:.4}", result.estimate);
    println!(
        "  Effective sample size: {:.1} / {}",
        result.effective_sample_size, result.n_samples
    );
    println!("  Weight variance: {:.4}", result.weight_variance);
    println!();

    // Likelihood weighting
    println!("Likelihood weighting for Bayesian networks:");
    println!();

    let net = sprinkler_network();
    let mut evidence = HashMap::new();
    evidence.insert(3, 1); // WetGrass = true

    // Exact answer for comparison
    let exact = net.query(2, &evidence).unwrap();

    // Approximate via likelihood weighting
    let lw_result = likelihood_weighting_with_diagnostics(&net, 2, &evidence, 10000, Some(42));

    println!("P(Rain | WetGrass=True):");
    println!("  Exact:       [{:.4}, {:.4}]", exact.p[0], exact.p[1]);
    println!(
        "  Approximate: [{:.4}, {:.4}]",
        lw_result.posterior.p[0], lw_result.posterior.p[1]
    );
    println!(
        "  ESS: {:.1} / {}",
        lw_result.effective_sample_size, lw_result.n_samples
    );
    println!();

    // -------------------------------------------------------------------------
    // 9. The Categorical Perspective
    // -------------------------------------------------------------------------
    println!("9. The Categorical Perspective");
    println!("------------------------------");
    println!();
    println!("Composition (applying a kernel) preserves normalization:");
    println!("  K: X → Dist(Y)  applied to P: Dist(X)  gives Q: Dist(Y)");
    println!();
    println!("Conditioning is NOT a morphism in FinStoch because:");
    println!("  1. It requires renormalization");
    println!("  2. It's undefined when P(E) = 0");
    println!("  3. It doesn't compose nicely with other operations");
    println!();
    println!("However, Bayes' rule can be expressed categorically via:");
    println!("  - Disintegration: P(X,Y) → (P(Y|X), P(X))");
    println!("  - Bayesian inversion: P(Y|X), P(X) → P(X|Y)");
    println!();

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    println!("=== Session 13 Complete ===");
    println!();
    println!("We implemented conditioning and Bayes' rule:");
    println!("  • bayes_update: P(H|E=e) via Bayes' rule");
    println!("  • sequential_update: Multiple evidence updates");
    println!("  • evidence_probability: The normalizing constant P(E)");
    println!("  • likelihood_ratio: For odds-based updates");
    println!("  • ExactInference: Queries on Bayesian networks");
    println!("  • mutual_information: Dependence between variables");
    println!("  • importance_sample: Approximate expectations via sampling");
    println!("  • likelihood_weighting: Approximate inference in Bayes nets");
    println!();
    println!("Key insight: Conditioning requires renormalization.");
    println!("It's fundamentally different from kernel composition.");
    println!();
    println!("When exact inference is intractable, importance sampling");
    println!("provides unbiased estimates by reweighting samples.");
    println!();
    println!("Next: Session 14 - Interventions and Do-Calculus");
}
