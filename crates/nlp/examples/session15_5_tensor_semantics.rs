//! Session 15.5: Tensor Semantics — From Grammar to Vectors
//!
//! Run with: cargo run -p compositional-nlp --example session15_5_tensor_semantics
//!
//! This example demonstrates:
//! - The semantics functor F: Grammar → Vect
//! - Word meanings as vectors and tensors
//! - Sentence meaning via tensor contraction
//! - How parse structure determines contractions
//!
//! Key insight: The pregroup parse tells us HOW to contract tensors!

use compositional_nlp::pregroup::{extract_cups, Grammar};
use compositional_nlp::semantics::{bilinear_form, inner_product, matrix_vector_mult, Semantics};

fn main() {
    println!("=== Session 15.5: Tensor Semantics ===\n");

    // -------------------------------------------------------------------------
    // 1. The Semantics Functor
    // -------------------------------------------------------------------------
    println!("1. The Semantics Functor F: Grammar → Vect");
    println!("------------------------------------------\n");

    println!("The functor maps:");
    println!("  F(N) = ℝⁿ           (noun space)");
    println!("  F(S) = ℝ            (sentence space = scalars)");
    println!("  F(Nˡ) = F(Nʳ) = ℝⁿ  (dual ≅ primal for finite dim)\n");

    println!("Morphism mapping (reductions → contractions):");
    println!("  N · Nʳ → 1  becomes inner product");
    println!("  Nˡ · N → 1  becomes inner product\n");

    // -------------------------------------------------------------------------
    // 2. Word Meanings as Tensors
    // -------------------------------------------------------------------------
    println!("2. Word Meanings as Tensors");
    println!("---------------------------\n");

    let semantics = Semantics::toy_lexicon();

    println!("Nouns → Vectors (in ℝ²):");
    println!("  Dimension 0: 'human-like' vs 'animal-like'");
    println!("  Dimension 1: 'active' vs 'passive'\n");

    let nouns = ["Alice", "Bob", "dog", "cat"];
    for noun in &nouns {
        if let Some(tensor) = semantics.get_tensor(noun) {
            println!("  {} ↦ {:?}", noun, tensor.as_vector().unwrap());
        }
    }
    println!();

    println!("Intransitive Verbs → Vectors (in ℝ²):");
    let verbs = ["runs", "sleeps", "walks"];
    for verb in &verbs {
        if let Some(tensor) = semantics.get_tensor(verb) {
            println!("  {} ↦ {:?}", verb, tensor.as_vector().unwrap());
        }
    }
    println!();

    println!("Transitive Verbs → Matrices (in ℝ²ˣ²):");
    let tv = ["loves", "sees", "chases"];
    for verb in &tv {
        if let Some(tensor) = semantics.get_tensor(verb) {
            println!("  {} ↦ {:?}", verb, tensor.as_matrix().unwrap());
        }
    }
    println!();

    println!("Adjectives → Matrices (linear maps ℝ² → ℝ²):");
    let adjs = ["big", "small", "happy"];
    for adj in &adjs {
        if let Some(tensor) = semantics.get_tensor(adj) {
            println!("  {} ↦ {:?}", adj, tensor.as_matrix().unwrap());
        }
    }
    println!();

    // -------------------------------------------------------------------------
    // 3. Example: "Alice runs" (Intransitive)
    // -------------------------------------------------------------------------
    println!("3. Example: \"Alice runs\" (Intransitive)");
    println!("----------------------------------------\n");

    let grammar = Grammar::english_basic();
    let parse = grammar.parse(&["Alice", "runs"]).unwrap();
    println!("{}", parse.trace());

    println!("Tensor contraction:");
    println!("  Alice = [0.9, 0.7]");
    println!("  runs  = [0.3, 0.95]");
    println!();

    let alice = vec![0.9, 0.7];
    let runs = vec![0.3, 0.95];
    let meaning = inner_product(&alice, &runs);
    println!("  ⟨Alice, runs⟩ = 0.9×0.3 + 0.7×0.95");
    println!("               = 0.27 + 0.665");
    println!("               = {:.4}\n", meaning);

    let computed = semantics.sentence_meaning(&parse).unwrap();
    println!("  Computed meaning: {:.4}", computed);
    println!();

    // -------------------------------------------------------------------------
    // 4. Example: "Alice loves Bob" (Transitive)
    // -------------------------------------------------------------------------
    println!("4. Example: \"Alice loves Bob\" (Transitive)");
    println!("-------------------------------------------\n");

    let parse = grammar.parse(&["Alice", "loves", "Bob"]).unwrap();
    println!("{}", parse.trace());

    // Show cups (contractions)
    let cups = extract_cups(&parse);
    println!("Cups (tensor contractions):");
    for cup in &cups {
        println!(
            "  Contract positions {} and {} (type {})",
            cup.left_pos, cup.right_pos, cup.base_type
        );
    }
    println!();

    println!("Tensor contraction:");
    println!("  Alice = [0.9, 0.7]");
    println!("  loves = [[0.9, 0.7],");
    println!("          [0.6, 0.8]]");
    println!("  Bob   = [0.8, 0.4]");
    println!();

    let alice = vec![0.9, 0.7];
    let loves = vec![vec![0.9, 0.7], vec![0.6, 0.8]];
    let bob = vec![0.8, 0.4];

    println!("  Step 1: loves · Bob = M · v");
    let loves_bob = matrix_vector_mult(&loves, &bob);
    println!("    [0.9×0.8 + 0.7×0.4, 0.6×0.8 + 0.8×0.4]");
    println!("    = [{:.2}, {:.2}]", loves_bob[0], loves_bob[1]);
    println!();

    println!("  Step 2: Alice · (loves · Bob) = ⟨v, w⟩");
    let meaning = inner_product(&alice, &loves_bob);
    println!(
        "    0.9×{:.2} + 0.7×{:.2} = {:.4}",
        loves_bob[0], loves_bob[1], meaning
    );
    println!();

    println!("  Or equivalently: Aliceᵀ · loves · Bob = {:.4}", meaning);
    let bilinear = bilinear_form(&alice, &loves, &bob);
    println!("  (bilinear form: {:.4})", bilinear);
    println!();

    let computed = semantics.sentence_meaning(&parse).unwrap();
    println!("  Computed meaning: {:.4}", computed);
    println!();

    // -------------------------------------------------------------------------
    // 5. Example: "big dog" (Adjective + Noun)
    // -------------------------------------------------------------------------
    println!("5. Example: \"big dog\" (Adjective + Noun)");
    println!("-----------------------------------------\n");

    let parse = grammar.parse(&["big", "dog"]).unwrap();
    println!("{}", parse.trace());

    println!("Tensor contraction:");
    println!("  big = [[1.0, 0.0],");
    println!("        [0.0, 1.2]]");
    println!("  dog = [0.2, 0.9]");
    println!();

    let big = vec![vec![1.0, 0.0], vec![0.0, 1.2]];
    let dog = vec![0.2, 0.9];
    let big_dog = matrix_vector_mult(&big, &dog);

    println!("  big · dog = A · v");
    println!("    [1.0×0.2 + 0.0×0.9, 0.0×0.2 + 1.2×0.9]");
    println!("    = [{:.2}, {:.2}]", big_dog[0], big_dog[1]);
    println!();

    println!("  The adjective 'big' increases the 'active' dimension!");
    println!();

    let np_vec = semantics.noun_phrase_meaning(&parse).unwrap();
    println!(
        "  Computed noun phrase: [{:.2}, {:.2}]",
        np_vec[0], np_vec[1]
    );
    println!();

    // -------------------------------------------------------------------------
    // 6. Comparing Sentence Meanings
    // -------------------------------------------------------------------------
    println!("6. Comparing Sentence Meanings");
    println!("------------------------------\n");

    let sentences = [
        ("Alice runs", vec!["Alice", "runs"]),
        ("Bob runs", vec!["Bob", "runs"]),
        ("dog runs", vec!["dog", "runs"]),
        ("Alice sleeps", vec!["Alice", "sleeps"]),
        ("dog sleeps", vec!["dog", "sleeps"]),
    ];

    println!("Sentence meanings (inner product of subject with verb):\n");
    for (sentence, words) in &sentences {
        let parse = grammar.parse(words).unwrap();
        let meaning = semantics.sentence_meaning(&parse).unwrap();
        println!("  \"{}\": {:.4}", sentence, meaning);
    }
    println!();

    println!("Observations:");
    println!("  - 'dog runs' has high meaning (dogs are active runners)");
    println!("  - 'Alice sleeps' has high meaning (humans sleep well)");
    println!("  - 'dog sleeps' has medium meaning (animals sleep less?)");
    println!();

    // -------------------------------------------------------------------------
    // 7. Asymmetry: "Alice loves Bob" vs "Bob loves Alice"
    // -------------------------------------------------------------------------
    println!("7. Asymmetry in Transitive Verbs");
    println!("---------------------------------\n");

    let parse1 = grammar.parse(&["Alice", "loves", "Bob"]).unwrap();
    let parse2 = grammar.parse(&["Bob", "loves", "Alice"]).unwrap();

    let m1 = semantics.sentence_meaning(&parse1).unwrap();
    let m2 = semantics.sentence_meaning(&parse2).unwrap();

    println!("  meaning(\"Alice loves Bob\") = {:.4}", m1);
    println!("  meaning(\"Bob loves Alice\") = {:.4}", m2);
    println!();
    println!("  Difference: {:.4}", (m1 - m2).abs());
    println!();
    println!("  The meanings differ because:");
    println!("    - Alice ≠ Bob (different vectors)");
    println!("    - The 'loves' matrix is not symmetric");
    println!("    - Word order matters through the bilinear form!");
    println!();

    // -------------------------------------------------------------------------
    // 8. Noun Phrase Similarity
    // -------------------------------------------------------------------------
    println!("8. Noun Phrase Similarity");
    println!("-------------------------\n");

    let phrases = [
        ("big dog", vec!["big", "dog"]),
        ("small dog", vec!["small", "dog"]),
        ("big cat", vec!["big", "cat"]),
        ("happy dog", vec!["happy", "dog"]),
    ];

    println!("Noun phrase vectors:\n");
    let mut np_vectors: Vec<(&str, Vec<f64>)> = Vec::new();
    for (phrase, words) in &phrases {
        let parse = grammar.parse(words).unwrap();
        let vec = semantics.noun_phrase_meaning(&parse).unwrap();
        println!("  \"{}\": [{:.3}, {:.3}]", phrase, vec[0], vec[1]);
        np_vectors.push((phrase, vec));
    }
    println!();

    println!("Cosine similarities:\n");
    for i in 0..np_vectors.len() {
        for j in (i + 1)..np_vectors.len() {
            let sim = Semantics::cosine_similarity(&np_vectors[i].1, &np_vectors[j].1);
            println!(
                "  \"{}\" vs \"{}\": {:.4}",
                np_vectors[i].0, np_vectors[j].0, sim
            );
        }
    }
    println!();

    // -------------------------------------------------------------------------
    // 9. String Diagram = Tensor Network
    // -------------------------------------------------------------------------
    println!("9. String Diagram = Tensor Network");
    println!("-----------------------------------\n");

    println!("For 'Alice loves Bob', the string diagram IS a tensor network:\n");
    println!("    Alice     loves        Bob");
    println!("      │      ┌──┴──┐        │");
    println!("     [v]    [M_ij]        [w]");
    println!("      │      /   \\         │");
    println!("      i     i     j        j");
    println!("      │     │     │        │");
    println!("      └─────┘     └────────┘");
    println!("         ↓            ↓");
    println!("       Σ_i         Σ_j");
    println!();
    println!("   Result: Σ_ij v_i · M_ij · w_j  (a scalar)");
    println!();

    // -------------------------------------------------------------------------
    // 10. The Functor Laws
    // -------------------------------------------------------------------------
    println!("10. The Functor Laws");
    println!("--------------------\n");

    println!("The semantics functor must preserve:");
    println!();
    println!("  Identity: F(id_X) = id_{{F(X)}}");
    println!("    - Identity type maps to identity matrix");
    println!();
    println!("  Composition: F(g ∘ f) = F(g) ∘ F(f)");
    println!("    - Sequential reductions = sequential contractions");
    println!();
    println!("This is what makes the semantics COMPOSITIONAL!");
    println!("The meaning of the whole is determined by the meanings of the parts.");
    println!();

    // -------------------------------------------------------------------------
    // Exercises
    // -------------------------------------------------------------------------
    println!("=== Exercises ===\n");

    println!("Exercise 1: Compute 'Alice loves Alice'");
    println!("---------------------------------------");
    let parse = grammar.parse(&["Alice", "loves", "Alice"]).unwrap();
    let meaning = semantics.sentence_meaning(&parse).unwrap();
    println!("  meaning = {:.4}", meaning);
    println!("  (Self-love using the bilinear form)\n");

    println!("Exercise 2: Identity matrix for 'loves'");
    println!("---------------------------------------");
    let mut sem2 = Semantics::new(2);
    sem2.add_noun("Alice", vec![1.0, 0.0]);
    sem2.add_noun("Bob", vec![0.0, 1.0]);
    sem2.add_transitive_verb("loves", vec![vec![1.0, 0.0], vec![0.0, 1.0]]); // Identity!

    let p1 = grammar.parse(&["Alice", "loves", "Bob"]).unwrap();
    let p2 = grammar.parse(&["Bob", "loves", "Alice"]).unwrap();
    let m1 = sem2.sentence_meaning(&p1).unwrap();
    let m2 = sem2.sentence_meaning(&p2).unwrap();
    println!("  With identity 'loves' matrix:");
    println!("    meaning(\"Alice loves Bob\") = {:.4}", m1);
    println!("    meaning(\"Bob loves Alice\") = {:.4}", m2);
    println!("  Both are 0 because Alice ⊥ Bob (orthogonal vectors)\n");

    println!("Exercise 3: Compose adjectives");
    println!("------------------------------");
    let parse = grammar.parse(&["big", "happy", "dog"]).unwrap();
    let np_vec = semantics.noun_phrase_meaning(&parse).unwrap();
    println!("  'big happy dog' = [{:.3}, {:.3}]", np_vec[0], np_vec[1]);
    println!("  (Both adjectives applied to dog vector)\n");

    println!("Exercise 4: Compare 'chases' meanings");
    println!("-------------------------------------");
    let p1 = grammar.parse(&["dog", "chases", "cat"]).unwrap();
    let p2 = grammar.parse(&["cat", "chases", "dog"]).unwrap();
    let m1 = semantics.sentence_meaning(&p1).unwrap();
    let m2 = semantics.sentence_meaning(&p2).unwrap();
    println!("  meaning(\"dog chases cat\") = {:.4}", m1);
    println!("  meaning(\"cat chases dog\") = {:.4}", m2);
    println!("  Dogs chase more than cats do!\n");

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    println!("=== Session 15.5 Complete ===\n");

    println!("We implemented the semantics functor F: Grammar → Vect:");
    println!("  • Nouns → vectors in ℝⁿ");
    println!("  • Transitive verbs → matrices in ℝⁿˣⁿ");
    println!("  • Intransitive verbs → vectors in ℝⁿ");
    println!("  • Adjectives → linear maps (matrices)");
    println!("  • Reductions (cups) → tensor contractions");
    println!();
    println!("Key insight: Parse structure determines contraction pattern!");
    println!();
    println!("The DisCoCat model combines:");
    println!("  - Distributional semantics (word vectors from data)");
    println!("  - Compositional structure (from grammar)");
    println!("  - Categorical coherence (functor laws)");
    println!();
    println!("Next: Session 16 — Full DisCoCat with similarity measures");
}
