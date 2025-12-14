//! Session 16: Full DisCoCat — Sentence Similarity and Applications
//!
//! Run with: cargo run -p compositional-nlp --example session16_discocat
//!
//! This example demonstrates the complete DisCoCat system:
//! - Sentence vectors (not just scalars)
//! - Similarity measures between sentences
//! - Verb tensor construction methods
//! - Comparison with baseline models (BoW, addition, multiplication)
//! - Evaluation on sentence similarity tasks
//!
//! Key insight: DisCoCat captures grammatical structure that baselines miss!

use compositional_nlp::discocat::{
    build_verb_tensor, contract_tensor3, cosine_similarity, norm, spearman_correlation, Baselines,
    DisCoCat, SentenceSemantics, VerbConstructionMethod,
};
use compositional_nlp::pregroup::Grammar;
use std::collections::HashMap;

fn main() {
    println!("=== Session 16: Full DisCoCat ===\n");

    // -------------------------------------------------------------------------
    // 1. Sentence Vectors
    // -------------------------------------------------------------------------
    println!("1. Sentence Vectors");
    println!("-------------------\n");

    println!("In Session 15.5, sentences mapped to scalars.");
    println!("Now we use: F(S) = ℝᵐ (sentence space)\n");

    let discocat = DisCoCat::with_toy_lexicon();

    println!("Example sentence vectors (4D):\n");

    let sentences = [
        vec!["Alice", "runs"],
        vec!["Bob", "runs"],
        vec!["dog", "runs"],
        vec!["Alice", "sleeps"],
        vec!["Alice", "loves", "Bob"],
        vec!["Bob", "loves", "Alice"],
    ];

    for sent in &sentences {
        if let Ok(vec) = discocat.sentence_vector(sent) {
            println!(
                "  \"{}\" → [{:.3}, {:.3}, {:.3}, {:.3}]",
                sent.join(" "),
                vec[0],
                vec[1],
                vec[2],
                vec[3]
            );
        }
    }
    println!();

    // -------------------------------------------------------------------------
    // 2. Sentence Similarity
    // -------------------------------------------------------------------------
    println!("2. Sentence Similarity");
    println!("----------------------\n");

    println!("Cosine similarity between sentence vectors:\n");

    let pairs = [
        (vec!["Alice", "runs"], vec!["Bob", "runs"]),
        (vec!["Alice", "runs"], vec!["Alice", "sleeps"]),
        (vec!["dog", "runs"], vec!["cat", "walks"]),
        (
            vec!["Alice", "loves", "Bob"],
            vec!["Alice", "adores", "Bob"],
        ),
        (vec!["Alice", "loves", "Bob"], vec!["Alice", "hates", "Bob"]),
        (vec!["Alice", "loves", "Bob"], vec!["Bob", "loves", "Alice"]),
    ];

    for (s1, s2) in &pairs {
        if let Ok(sim) = discocat.similarity(s1, s2) {
            println!("  \"{}\" vs \"{}\": {:.4}", s1.join(" "), s2.join(" "), sim);
        }
    }
    println!();

    println!("Observations:");
    println!("  - Similar activities (runs) have similar meanings");
    println!("  - 'loves' is closer to 'adores' than to 'hates'");
    println!("  - Subject/object swap changes the meaning!");
    println!();

    // -------------------------------------------------------------------------
    // 3. Word Order Matters: DisCoCat vs Bag-of-Words
    // -------------------------------------------------------------------------
    println!("3. Word Order Matters");
    println!("---------------------\n");

    println!("The classic example: 'dog bites man' vs 'man bites dog'\n");

    let s1 = vec!["dog", "bites", "man"];
    let s2 = vec!["man", "bites", "dog"];

    let discocat_sim = discocat.similarity(&s1, &s2).unwrap_or(0.0);
    let bow_sim = discocat.bow_similarity(&s1, &s2);
    let add_sim = discocat.add_similarity(&s1, &s2);

    println!("  DisCoCat similarity: {:.4}", discocat_sim);
    println!("  Bag-of-Words similarity: {:.4}", bow_sim);
    println!("  Addition model similarity: {:.4}", add_sim);
    println!();

    println!("Analysis:");
    println!("  BoW = {:.4} because both have the same words!", bow_sim);
    println!(
        "  DisCoCat = {:.4} because it uses grammatical structure",
        discocat_sim
    );
    println!("  The tensor contraction distinguishes subject from object.");
    println!();

    // Show the actual vectors
    if let (Ok(v1), Ok(v2)) = (discocat.sentence_vector(&s1), discocat.sentence_vector(&s2)) {
        println!(
            "  \"dog bites man\" → [{:.3}, {:.3}, {:.3}, {:.3}]",
            v1[0], v1[1], v1[2], v1[3]
        );
        println!(
            "  \"man bites dog\" → [{:.3}, {:.3}, {:.3}, {:.3}]",
            v2[0], v2[1], v2[2], v2[3]
        );
    }
    println!();

    // -------------------------------------------------------------------------
    // 4. Baseline Model Comparison
    // -------------------------------------------------------------------------
    println!("4. Baseline Model Comparison");
    println!("----------------------------\n");

    // Create a simple lexicon for baselines
    let mut baseline_lexicon: HashMap<String, Vec<f64>> = HashMap::new();
    baseline_lexicon.insert("Alice".to_string(), vec![0.9, 0.1, 0.7, 0.8]);
    baseline_lexicon.insert("Bob".to_string(), vec![0.85, 0.1, 0.5, 0.6]);
    baseline_lexicon.insert("dog".to_string(), vec![0.1, 0.9, 0.9, 0.7]);
    baseline_lexicon.insert("cat".to_string(), vec![0.1, 0.85, 0.6, 0.6]);
    baseline_lexicon.insert("man".to_string(), vec![0.95, 0.05, 0.6, 0.5]);
    baseline_lexicon.insert("loves".to_string(), vec![0.5, 0.5, 0.4, 0.9]);
    baseline_lexicon.insert("hates".to_string(), vec![0.5, 0.5, 0.4, 0.1]);
    baseline_lexicon.insert("bites".to_string(), vec![0.3, 0.7, 0.8, 0.2]);
    baseline_lexicon.insert("runs".to_string(), vec![0.4, 0.6, 0.9, 0.6]);

    println!("Comparing models on 'Alice loves Bob' vs 'Alice hates Bob':\n");

    let s1 = vec!["Alice", "loves", "Bob"];
    let s2 = vec!["Alice", "hates", "Bob"];

    println!("  Model        | Similarity");
    println!("  -------------|----------");
    println!(
        "  DisCoCat     | {:.4}",
        discocat.similarity(&s1, &s2).unwrap_or(0.0)
    );
    println!(
        "  Bag-of-Words | {:.4}",
        Baselines::bow_similarity(&s1, &s2, &baseline_lexicon)
    );
    println!(
        "  Addition     | {:.4}",
        Baselines::add_similarity(&s1, &s2, &baseline_lexicon)
    );
    println!(
        "  Multiply     | {:.4}",
        Baselines::mult_similarity(&s1, &s2, &baseline_lexicon)
    );
    println!();

    println!("Key differences:");
    println!("  - BoW/Addition give high similarity (same subject/object)");
    println!("  - DisCoCat captures the verb difference through tensor structure");
    println!();

    // -------------------------------------------------------------------------
    // 5. Verb Tensor Construction
    // -------------------------------------------------------------------------
    println!("5. Verb Tensor Construction");
    println!("---------------------------\n");

    println!("Methods for building verb tensors from data:\n");

    // Example corpus for "chases"
    let dog = vec![0.1, 0.9, 0.9, 0.7];
    let cat = vec![0.1, 0.85, 0.6, 0.6];
    let mouse = vec![0.05, 0.8, 0.7, 0.4];

    let pairs: Vec<(&[f64], &[f64])> = vec![
        (dog.as_slice(), cat.as_slice()),
        (cat.as_slice(), mouse.as_slice()),
        (dog.as_slice(), mouse.as_slice()),
    ];

    println!("Corpus for 'chases': (dog, cat), (cat, mouse), (dog, mouse)\n");

    // Build tensors with different methods
    let relational = build_verb_tensor(&pairs, 4, 4, VerbConstructionMethod::Relational);
    let kronecker = build_verb_tensor(&pairs, 4, 4, VerbConstructionMethod::Kronecker);
    let copy_subj = build_verb_tensor(&pairs, 4, 4, VerbConstructionMethod::CopySubject);

    // Test with dog chases cat
    let sent_rel = contract_tensor3(&relational, &dog, &cat);
    let sent_kron = contract_tensor3(&kronecker, &dog, &cat);
    let sent_copy = contract_tensor3(&copy_subj, &dog, &cat);

    println!("Sentence vector for 'dog chases cat' with each method:\n");
    println!(
        "  Relational:   [{:.3}, {:.3}, {:.3}, {:.3}] (norm: {:.3})",
        sent_rel[0],
        sent_rel[1],
        sent_rel[2],
        sent_rel[3],
        norm(&sent_rel)
    );
    println!(
        "  Kronecker:    [{:.3}, {:.3}, {:.3}, {:.3}] (norm: {:.3})",
        sent_kron[0],
        sent_kron[1],
        sent_kron[2],
        sent_kron[3],
        norm(&sent_kron)
    );
    println!(
        "  Copy-Subject: [{:.3}, {:.3}, {:.3}, {:.3}] (norm: {:.3})",
        sent_copy[0],
        sent_copy[1],
        sent_copy[2],
        sent_copy[3],
        norm(&sent_copy)
    );
    println!();

    // -------------------------------------------------------------------------
    // 6. Intransitive vs Transitive
    // -------------------------------------------------------------------------
    println!("6. Intransitive vs Transitive Verbs");
    println!("------------------------------------\n");

    println!("Intransitive: N · (Nʳ·S) → S via matrix multiplication");
    println!("Transitive:   N · (Nʳ·S·Nˡ) · N → S via tensor contraction\n");

    let grammar = Grammar::english_basic();

    let intrans = grammar.parse(&["Alice", "runs"]).unwrap();
    let trans = grammar.parse(&["Alice", "loves", "Bob"]).unwrap();

    println!("Parse: 'Alice runs'");
    println!("{}", intrans.trace());

    println!("Parse: 'Alice loves Bob'");
    println!("{}", trans.trace());

    // -------------------------------------------------------------------------
    // 7. Similarity Evaluation
    // -------------------------------------------------------------------------
    println!("7. Similarity Evaluation");
    println!("------------------------\n");

    // Create test dataset with gold similarities
    let test_pairs: Vec<(Vec<&str>, Vec<&str>, f64)> = vec![
        (vec!["Alice", "runs"], vec!["Bob", "runs"], 0.8),
        (vec!["Alice", "runs"], vec!["Alice", "walks"], 0.7),
        (vec!["Alice", "runs"], vec!["Alice", "sleeps"], 0.4),
        (vec!["dog", "runs"], vec!["cat", "runs"], 0.75),
        (
            vec!["Alice", "loves", "Bob"],
            vec!["Alice", "adores", "Bob"],
            0.9,
        ),
        (
            vec!["Alice", "loves", "Bob"],
            vec!["Alice", "sees", "Bob"],
            0.5,
        ),
        (
            vec!["Alice", "loves", "Bob"],
            vec!["Alice", "hates", "Bob"],
            0.2,
        ),
        (
            vec!["dog", "chases", "cat"],
            vec!["cat", "chases", "mouse"],
            0.6,
        ),
    ];

    println!(
        "Evaluating on {} sentence pairs with gold ratings:\n",
        test_pairs.len()
    );

    let result = discocat.evaluate_similarity(&test_pairs);
    println!("{}", result);

    println!("Interpretation:");
    println!("  ρ = 1.0: perfect correlation with human judgments");
    println!("  ρ = 0.0: no correlation");
    println!("  ρ = -1.0: inverse correlation");
    println!();

    // -------------------------------------------------------------------------
    // 8. Entailment (Simple)
    // -------------------------------------------------------------------------
    println!("8. Textual Entailment (Simple)");
    println!("------------------------------\n");

    println!("Simple threshold-based entailment check:\n");

    let entailment_pairs = [
        (vec!["dog", "runs"], vec!["dog", "walks"]),
        (
            vec!["Alice", "adores", "Bob"],
            vec!["Alice", "loves", "Bob"],
        ),
        (vec!["cat", "sleeps"], vec!["dog", "runs"]),
    ];

    for (premise, hypothesis) in &entailment_pairs {
        let sim = discocat.similarity(premise, hypothesis).unwrap_or(0.0);
        let entails = discocat.entails(premise, hypothesis).unwrap_or(false);
        println!("  \"{}\" → \"{}\"", premise.join(" "), hypothesis.join(" "));
        println!("    Similarity: {:.4}, Entails: {}\n", sim, entails);
    }

    // -------------------------------------------------------------------------
    // 9. String Diagram = Tensor Network
    // -------------------------------------------------------------------------
    println!("9. String Diagram = Tensor Network");
    println!("-----------------------------------\n");

    println!("For 'Alice loves Bob' with sentence vectors:\n");
    println!("       Alice      loves            Bob");
    println!("         │      ┌──┴──┐            │");
    println!("        vᵢ     Tᵢⱼₖ             wₖ");
    println!("         │     /    \\             │");
    println!("         └────┘      └────────────┘");
    println!("            ↓              ↓");
    println!("          Σᵢ            Σₖ");
    println!("                ↓");
    println!("            sⱼ ∈ ℝ⁴  (sentence vector)");
    println!();
    println!("Formula: sⱼ = Σᵢₖ vᵢ · Tᵢⱼₖ · wₖ");
    println!();

    // -------------------------------------------------------------------------
    // 10. Noun Phrase Similarity
    // -------------------------------------------------------------------------
    println!("10. Noun Phrase Similarity");
    println!("--------------------------\n");

    let semantics = SentenceSemantics::toy_lexicon();
    let grammar = Grammar::english_basic();

    let phrases = [
        vec!["big", "dog"],
        vec!["small", "dog"],
        vec!["big", "cat"],
        vec!["happy", "dog"],
    ];

    println!("Noun phrase vectors:\n");
    let mut np_vecs: Vec<(String, Vec<f64>)> = Vec::new();

    for phrase in &phrases {
        if let Ok(parse) = grammar.parse(phrase) {
            if let Ok(vec) = semantics.noun_phrase_vector(&parse) {
                println!(
                    "  \"{}\" → [{:.3}, {:.3}, {:.3}, {:.3}]",
                    phrase.join(" "),
                    vec[0],
                    vec[1],
                    vec[2],
                    vec[3]
                );
                np_vecs.push((phrase.join(" "), vec));
            }
        }
    }
    println!();

    println!("Cosine similarities:\n");
    for i in 0..np_vecs.len() {
        for j in (i + 1)..np_vecs.len() {
            let sim = cosine_similarity(&np_vecs[i].1, &np_vecs[j].1);
            println!("  \"{}\" vs \"{}\": {:.4}", np_vecs[i].0, np_vecs[j].0, sim);
        }
    }
    println!();

    // -------------------------------------------------------------------------
    // Exercises
    // -------------------------------------------------------------------------
    println!("=== Exercises ===\n");

    println!("Exercise 1: Compare verb similarities");
    println!("--------------------------------------");
    let verbs = [
        (
            vec!["Alice", "loves", "Bob"],
            vec!["Alice", "adores", "Bob"],
        ),
        (vec!["Alice", "loves", "Bob"], vec!["Alice", "sees", "Bob"]),
        (vec!["dog", "chases", "cat"], vec!["dog", "bites", "cat"]),
    ];
    for (s1, s2) in &verbs {
        if let Ok(sim) = discocat.similarity(s1, s2) {
            println!("  \"{}\" vs \"{}\": {:.4}", s1.join(" "), s2.join(" "), sim);
        }
    }
    println!();

    println!("Exercise 2: Subject/object asymmetry");
    println!("-------------------------------------");
    let asymmetric = [
        (vec!["Alice", "loves", "Bob"], vec!["Bob", "loves", "Alice"]),
        (vec!["dog", "chases", "cat"], vec!["cat", "chases", "dog"]),
    ];
    for (s1, s2) in &asymmetric {
        if let Ok(sim) = discocat.similarity(s1, s2) {
            println!("  \"{}\" vs \"{}\": {:.4}", s1.join(" "), s2.join(" "), sim);
        }
    }
    println!();

    println!("Exercise 3: Spearman correlation example");
    println!("-----------------------------------------");
    let predictions = vec![0.9, 0.7, 0.5, 0.3, 0.1];
    let gold = vec![0.95, 0.75, 0.45, 0.35, 0.15];
    let rho = spearman_correlation(&predictions, &gold);
    println!("  Predictions: {:?}", predictions);
    println!("  Gold:        {:?}", gold);
    println!("  Spearman ρ:  {:.4}", rho);
    println!();

    println!("Exercise 4: Build your own verb tensor");
    println!("--------------------------------------");
    // Using copy-subject for a simple "likes" verb
    let likes_tensor = build_verb_tensor(&[], 4, 4, VerbConstructionMethod::CopySubject);
    let alice = vec![0.9, 0.1, 0.7, 0.8];
    let bob = vec![0.85, 0.1, 0.5, 0.6];
    let sent = contract_tensor3(&likes_tensor, &alice, &bob);
    println!("  'likes' with CopySubject method:");
    println!(
        "  Alice likes Bob → [{:.3}, {:.3}, {:.3}, {:.3}]",
        sent[0], sent[1], sent[2], sent[3]
    );
    println!("  (Diagonal structure: meaning depends on subject-object alignment)");
    println!();

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    println!("=== Session 16 Complete ===\n");

    println!("We implemented the full DisCoCat system:");
    println!("  • Sentence vectors in ℝᵐ (not just scalars)");
    println!("  • Transitive verbs as order-3 tensors");
    println!("  • Cosine similarity for sentence comparison");
    println!("  • Verb tensor construction (relational, Kronecker, copy)");
    println!("  • Baseline models (BoW, addition, multiplication)");
    println!("  • Spearman correlation for evaluation");
    println!();
    println!("Key insights:");
    println!("  • DisCoCat captures word order through tensor structure");
    println!("  • Baselines (BoW) miss grammatical distinctions");
    println!("  • The functor F: Grammar → Vect is compositional");
    println!("  • Evaluation correlates with human similarity judgments");
    println!();
    println!("The DisCoCat framework unifies:");
    println!("  - Distributional semantics (word vectors from data)");
    println!("  - Compositional structure (from pregroup grammar)");
    println!("  - Categorical coherence (functor laws)");
    println!();
    println!("Next: Session 17 — Monoidal categories and string diagrams");
}
