//! Session 14: Interventions and Do-Calculus
//!
//! Run with: cargo run -p compositional-prob --example session14_interventions
//!
//! This example demonstrates:
//! - Why do(X=x) ≠ observing X=x
//! - Graph surgery for interventions
//! - The backdoor adjustment formula
//! - Confounding and its effects
//! - Average Treatment Effect (ATE)
//!
//! Key insight: Intervening breaks causal mechanisms, observing doesn't!

use compositional_prob::{
    causal::{
        average_treatment_effect, backdoor_adjustment, causal_effect, confounding_bias,
        interventional_query,
    },
    smoking_cancer_network, sprinkler_network, Dist, Kernel,
};
use std::collections::HashMap;

fn main() {
    println!("=== Session 14: Interventions and Do-Calculus ===\n");

    // -------------------------------------------------------------------------
    // 1. The Core Insight: do(X=x) ≠ observing X=x
    // -------------------------------------------------------------------------
    println!("1. The Core Insight: do(X=x) ≠ observing X=x");
    println!("----------------------------------------------");
    println!();
    println!("Observing X=x: We learn X=x, which tells us about X's causes");
    println!("Intervening do(X=x): We force X=x, breaking the causal mechanism");
    println!();

    let net = sprinkler_network();

    // Observational: P(Cloudy | Sprinkler=On)
    let mut evidence = HashMap::new();
    evidence.insert(1, 1); // Sprinkler = On
    let p_cloudy_obs = net.query(0, &evidence).unwrap();

    // Interventional: P(Cloudy | do(Sprinkler=On))
    let intervened = net.intervene(1, 1);
    let p_cloudy_int = intervened.marginal(0).unwrap();

    println!("Sprinkler Network Example:");
    println!(
        "  P(Cloudy | Sprinkler=On)    = [{:.4}, {:.4}]",
        p_cloudy_obs.p[0], p_cloudy_obs.p[1]
    );
    println!(
        "  P(Cloudy | do(Sprinkler=On)) = [{:.4}, {:.4}]",
        p_cloudy_int.p[0], p_cloudy_int.p[1]
    );
    println!();
    println!("Why the difference?");
    println!("  - Observing Sprinkler=On tells us it's probably NOT cloudy");
    println!("    (since sprinklers run more when it's sunny)");
    println!("  - Intervening do(Sprinkler=On) doesn't change Cloudy at all");
    println!("    (we're forcing the sprinkler, not learning about weather)");
    println!();

    // -------------------------------------------------------------------------
    // 2. Graph Surgery: The do-Operator
    // -------------------------------------------------------------------------
    println!("2. Graph Surgery: The do-Operator");
    println!("----------------------------------");
    println!();
    println!("When we intervene do(X=x):");
    println!("  1. Remove all edges INTO X (cut the causal mechanism)");
    println!("  2. Replace P(X|parents) with a point mass at x");
    println!();
    println!("Original:     Cloudy → Sprinkler → WetGrass");
    println!("                   ↘      ↗");
    println!("                   Rain");
    println!();
    println!("do(Sprinkler=On): Cloudy    Sprinkler=On → WetGrass");
    println!("                       ↘         ↗");
    println!("                       Rain");
    println!();

    // Show the effect on downstream variables
    let p_wet_obs = {
        let mut ev = HashMap::new();
        ev.insert(1, 1);
        net.query(3, &ev).unwrap()
    };
    let p_wet_int = intervened.marginal(3).unwrap();

    println!("Effect on WetGrass:");
    println!(
        "  P(WetGrass | Sprinkler=On)    = [{:.4}, {:.4}]",
        p_wet_obs.p[0], p_wet_obs.p[1]
    );
    println!(
        "  P(WetGrass | do(Sprinkler=On)) = [{:.4}, {:.4}]",
        p_wet_int.p[0], p_wet_int.p[1]
    );
    println!();

    // -------------------------------------------------------------------------
    // 3. Confounding: The Smoking Example
    // -------------------------------------------------------------------------
    println!("3. Confounding: The Smoking Example");
    println!("------------------------------------");
    println!();
    println!("Causal structure:");
    println!("       Genotype (G)");
    println!("       ↙        ↘");
    println!("  Smoking (S) → Tar (T) → Cancer (C)");
    println!();
    println!("G is a CONFOUNDER: it affects both Smoking and Cancer");
    println!("This creates a spurious association beyond the causal effect.");
    println!();

    let smoking_net = smoking_cancer_network();

    // Observational: P(Cancer | Smoking=1)
    let mut smoke_evidence = HashMap::new();
    smoke_evidence.insert(1, 1);
    let p_cancer_obs = smoking_net.query(3, &smoke_evidence).unwrap();

    // Interventional: P(Cancer | do(Smoking=1))
    let p_cancer_int = interventional_query(&smoking_net, 3, 1, 1).unwrap();

    println!("Comparing observational and interventional:");
    println!(
        "  P(Cancer=1 | Smoking=1)     = {:.4} (observational)",
        p_cancer_obs.p[1]
    );
    println!(
        "  P(Cancer=1 | do(Smoking=1)) = {:.4} (interventional)",
        p_cancer_int.p[1]
    );
    println!();
    println!("The observational estimate is HIGHER because:");
    println!("  - People who smoke tend to have the 'smoking gene'");
    println!("  - The gene independently increases cancer risk");
    println!("  - So observing someone smokes tells us they likely have the gene");
    println!();

    let bias = confounding_bias(&smoking_net, 1, 1, 3).unwrap();
    println!("Confounding bias: {:.4}", bias);
    println!("  (difference between observational and causal effect)");
    println!();

    // -------------------------------------------------------------------------
    // 4. The Backdoor Adjustment Formula
    // -------------------------------------------------------------------------
    println!("4. The Backdoor Adjustment Formula");
    println!("-----------------------------------");
    println!();
    println!("If we can't intervene, but we observe the confounder Z:");
    println!();
    println!("  P(Y | do(X=x)) = Σ_z P(Y | X=x, Z=z) · P(Z=z)");
    println!();
    println!("This 'adjusts' for the confounder by averaging over its distribution.");
    println!();

    // Backdoor adjustment using Genotype
    let p_cancer_adj = backdoor_adjustment(&smoking_net, 1, 1, 3, &[0]).unwrap();

    println!("Adjusting for Genotype:");
    println!(
        "  P(Cancer=1 | do(Smoking=1)) via intervention = {:.4}",
        p_cancer_int.p[1]
    );
    println!(
        "  P(Cancer=1 | do(Smoking=1)) via adjustment   = {:.4}",
        p_cancer_adj.p[1]
    );
    println!();
    println!("The adjustment formula gives the same answer as intervention!");
    println!("This is how we estimate causal effects from observational data.");
    println!();

    // -------------------------------------------------------------------------
    // 5. Average Treatment Effect (ATE)
    // -------------------------------------------------------------------------
    println!("5. Average Treatment Effect (ATE)");
    println!("---------------------------------");
    println!();
    println!("ATE = E[Y | do(X=1)] - E[Y | do(X=0)]");
    println!();
    println!("For binary outcomes:");
    println!("  ATE = P(Y=1 | do(X=1)) - P(Y=1 | do(X=0))");
    println!();

    let effect = causal_effect(&smoking_net, 1, 3).unwrap();

    println!("Causal effect of Smoking on Cancer:");
    println!(
        "  P(Cancer=1 | do(Smoking=0)) = {:.4} (baseline risk)",
        effect.risk_control
    );
    println!(
        "  P(Cancer=1 | do(Smoking=1)) = {:.4} (risk if everyone smokes)",
        effect.risk_treated
    );
    println!("  ATE = {:.4}", effect.ate);
    println!("  Relative Risk = {:.2}", effect.relative_risk);
    println!();
    println!(
        "Interpretation: Smoking increases cancer risk by {:.1} percentage points",
        effect.ate * 100.0
    );
    println!();

    // -------------------------------------------------------------------------
    // 6. Simple Treatment Model
    // -------------------------------------------------------------------------
    println!("6. Simple Treatment Model");
    println!("-------------------------");
    println!();

    // Create a simple treatment -> outcome model
    use compositional_prob::BayesNet;
    let mut simple_net = BayesNet::new(vec![2, 2]); // Treatment, Outcome
    simple_net
        .add_prior(0, &Dist::new(vec![0.5, 0.5]).unwrap())
        .unwrap();
    simple_net
        .add_conditional(
            1,
            vec![0],
            Kernel::new(vec![
                vec![0.8, 0.2], // No treatment: 20% positive outcome
                vec![0.3, 0.7], // Treatment: 70% positive outcome
            ])
            .unwrap(),
        )
        .unwrap();

    let simple_ate = average_treatment_effect(&simple_net, 0, 1).unwrap();

    println!("Simple model: Treatment → Outcome");
    println!("  P(Outcome=1 | no treatment) = 0.2");
    println!("  P(Outcome=1 | treatment)    = 0.7");
    println!("  ATE = {:.2}", simple_ate);
    println!();
    println!("No confounding, so observational = interventional!");
    println!();

    // -------------------------------------------------------------------------
    // 7. The Categorical Perspective
    // -------------------------------------------------------------------------
    println!("7. The Categorical Perspective");
    println!("------------------------------");
    println!();
    println!("In categorical terms, an intervention is a functor that:");
    println!("  1. Removes a morphism (the mechanism for X)");
    println!("  2. Replaces it with a constant morphism (point mass)");
    println!();
    println!("Original network as composed morphisms:");
    println!("  P(G) ; P(S|G) ; P(T|S) ; P(C|G,T)");
    println!();
    println!("After do(S=1):");
    println!("  P(G) ; δ₁ ; P(T|S=1) ; P(C|G,T)");
    println!();
    println!("where δ₁ is the point mass at S=1.");
    println!();

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    println!("=== Session 14 Complete ===");
    println!();
    println!("We implemented interventions and causal inference:");
    println!("  • BayesNet::intervene: Graph surgery for do(X=x)");
    println!("  • interventional_query: P(Y | do(X=x))");
    println!("  • backdoor_adjustment: Causal effect from observations");
    println!("  • average_treatment_effect: ATE estimation");
    println!("  • confounding_bias: Difference between obs and causal");
    println!();
    println!("Key insight: do(X=x) ≠ observing X=x");
    println!("Intervening breaks causal mechanisms; observing doesn't.");
    println!();
    println!("Next: Session 15 - Counterfactuals");
}
