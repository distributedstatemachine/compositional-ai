//! Session 15: Pregroup Grammars and Type Reductions
//!
//! Run with: cargo run -p compositional-nlp --example session15_pregroup
//!
//! This example demonstrates:
//! - Pregroup type system (N, S, Nʳ, Nˡ, etc.)
//! - Type reduction rules (X · Xʳ → 1, Xˡ · X → 1)
//! - Grammaticality as type reduction to S
//! - Various sentence structures (intransitive, transitive, complex)
//! - Why word order matters
//!
//! Key insight: Reductions tell us which wires to contract in tensor networks!

use compositional_nlp::pregroup::{extract_cups, AtomicType, BasicType, Grammar, PregroupType};

fn main() {
    println!("=== Session 15: Pregroup Grammars and Type Reductions ===\n");

    // -------------------------------------------------------------------------
    // 1. Basic Pregroup Types
    // -------------------------------------------------------------------------
    println!("1. Basic Pregroup Types");
    println!("-----------------------\n");

    println!("Basic types:");
    println!("  N  = noun");
    println!("  S  = sentence\n");

    println!("Adjoint types:");
    let n = AtomicType::basic(BasicType::N);
    let nr = n.right_adjoint();
    let nl = n.left_adjoint();

    println!("  {} = noun", n);
    println!("  {} = right adjoint (expects N on right)", nr);
    println!("  {} = left adjoint (expects N on left)", nl);
    println!();

    // -------------------------------------------------------------------------
    // 2. Word Types
    // -------------------------------------------------------------------------
    println!("2. Word Types");
    println!("-------------\n");

    println!("Common word type assignments:");
    println!();

    let noun = PregroupType::noun();
    println!("  Noun (Alice, Bob, dog): {}", noun);

    let iv = PregroupType::intransitive_verb();
    println!("  Intransitive verb (runs, sleeps): {}", iv);
    println!("    Takes subject on left, produces sentence");

    let tv = PregroupType::transitive_verb();
    println!("  Transitive verb (loves, sees): {}", tv);
    println!("    Takes subject on left, object on right");

    let adj = PregroupType::adjective();
    println!("  Adjective (big, red): {}", adj);
    println!("    Modifies noun on right");

    let det = PregroupType::determiner();
    println!("  Determiner (the, a): {}", det);
    println!("    Turns noun into noun phrase");
    println!();

    // -------------------------------------------------------------------------
    // 3. Reduction Rules
    // -------------------------------------------------------------------------
    println!("3. Reduction Rules");
    println!("------------------\n");

    println!("The key insight: types 'cancel' like fractions!\n");

    println!("Right reduction: X · Xʳ → 1");
    println!("  N · Nʳ → 1 (noun followed by 'wants noun on right')");
    println!();

    println!("Left reduction: Xˡ · X → 1");
    println!("  Nˡ · N → 1 ('wants noun on left' followed by noun)");
    println!();

    // -------------------------------------------------------------------------
    // 4. Example: "Alice runs" (Intransitive)
    // -------------------------------------------------------------------------
    println!("4. Example: \"Alice runs\" (Intransitive)");
    println!("----------------------------------------\n");

    let grammar = Grammar::english_basic();
    let result = grammar.parse(&["Alice", "runs"]).unwrap();

    println!("{}", result.trace());

    // -------------------------------------------------------------------------
    // 5. Example: "Alice loves Bob" (Transitive)
    // -------------------------------------------------------------------------
    println!("\n5. Example: \"Alice loves Bob\" (Transitive)");
    println!("-------------------------------------------\n");

    let result = grammar.parse(&["Alice", "loves", "Bob"]).unwrap();
    println!("{}", result.trace());

    // Show cups
    let cups = extract_cups(&result);
    println!("Cups (wire contractions):");
    for cup in &cups {
        println!(
            "  Contract positions {} and {} (type {})",
            cup.left_pos, cup.right_pos, cup.base_type
        );
    }
    println!();

    // -------------------------------------------------------------------------
    // 6. Example: "the big dog runs" (Complex)
    // -------------------------------------------------------------------------
    println!("6. Example: \"the big dog runs\" (Complex)");
    println!("-----------------------------------------\n");

    let result = grammar.parse(&["the", "big", "dog", "runs"]).unwrap();
    println!("{}", result.trace());

    // -------------------------------------------------------------------------
    // 7. Example: "Alice sees the big cat"
    // -------------------------------------------------------------------------
    println!("\n7. Example: \"Alice sees the big cat\"");
    println!("-------------------------------------\n");

    let result = grammar
        .parse(&["Alice", "sees", "the", "big", "cat"])
        .unwrap();
    println!("{}", result.trace());

    // -------------------------------------------------------------------------
    // 8. Example: "Alice thinks Bob runs" (Sentence Complement)
    // -------------------------------------------------------------------------
    println!("\n8. Example: \"Alice thinks Bob runs\" (Sentence Complement)");
    println!("----------------------------------------------------------\n");

    let result = grammar.parse(&["Alice", "thinks", "Bob", "runs"]).unwrap();
    println!("{}", result.trace());

    // -------------------------------------------------------------------------
    // 9. Ungrammatical: "loves Alice Bob"
    // -------------------------------------------------------------------------
    println!("\n9. Ungrammatical: \"loves Alice Bob\"");
    println!("------------------------------------\n");

    let result = grammar.parse(&["loves", "Alice", "Bob"]).unwrap();
    println!("{}", result.trace());

    println!("Why ungrammatical?");
    println!("  loves : Nʳ · S · Nˡ");
    println!("  The Nʳ at the start has nothing on its left to reduce with!");
    println!();

    // -------------------------------------------------------------------------
    // 10. Noun Phrases
    // -------------------------------------------------------------------------
    println!("10. Noun Phrases (reduce to N, not S)");
    println!("-------------------------------------\n");

    let result = grammar.parse(&["the", "big", "dog"]).unwrap();
    println!("{}", result.trace());

    // -------------------------------------------------------------------------
    // 11. String Diagram View
    // -------------------------------------------------------------------------
    println!("\n11. String Diagram View");
    println!("-----------------------\n");

    println!("For 'Alice loves Bob':");
    println!();
    println!("  Alice    loves       Bob");
    println!("    │    ┌──┴──┐        │");
    println!("    N    Nʳ  S  Nˡ      N");
    println!("    │    │      │       │");
    println!("    └────┘      └───────┘");
    println!("         ↓           ↓");
    println!("         1           1");
    println!();
    println!("         S ← Final output");
    println!();
    println!("The cups (curved lines) represent type reductions.");
    println!("In a tensor network, these are index contractions!");
    println!();

    // -------------------------------------------------------------------------
    // 12. Connection to DisCoCat
    // -------------------------------------------------------------------------
    println!("12. Connection to DisCoCat");
    println!("--------------------------\n");

    println!("DisCoCat (Distributional Compositional Categorical) uses:");
    println!();
    println!("  1. PARSE: Get grammatical structure (pregroup reductions)");
    println!("  2. ASSIGN: Give vectors/tensors to words");
    println!("     - Nouns: vectors in ℝⁿ");
    println!("     - Transitive verbs: matrices in ℝⁿ ⊗ ℝⁿ");
    println!("  3. CONTRACT: Use cups as tensor contractions");
    println!("  4. RESULT: Sentence meaning as a scalar/vector");
    println!();
    println!("The pregroup parse tells us HOW to contract!");
    println!();

    // -------------------------------------------------------------------------
    // Exercises
    // -------------------------------------------------------------------------
    println!("=== Exercises ===\n");

    println!("Exercise 1: Parse 'the dog runs'");
    println!("---------------------------------");
    let result = grammar.parse(&["the", "dog", "runs"]).unwrap();
    println!("{}", result.trace());

    println!("\nExercise 2: Why is 'dog the runs' ungrammatical?");
    println!("------------------------------------------------");
    let result = grammar.parse(&["dog", "the", "runs"]).unwrap();
    println!("{}", result.trace());
    println!(
        "The determiner 'the' expects a noun on its right (Nˡ),\nbut 'runs' is Nʳ · S, not N!"
    );
    println!();

    println!("Exercise 3: Parse 'Alice sees Bob'");
    println!("----------------------------------");
    let result = grammar.parse(&["Alice", "sees", "Bob"]).unwrap();
    println!("{}", result.trace());

    println!("\nExercise 4: Build 'big' type");
    println!("----------------------------");
    let big_type = PregroupType::adjective();
    println!("'big' has type: {}", big_type);
    println!("This is N · Nˡ because:");
    println!("  - It outputs N (it's a noun modifier)");
    println!("  - It consumes N on its right (the noun it modifies)");
    println!();

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    println!("=== Session 15 Complete ===\n");

    println!("We implemented pregroup grammars:");
    println!("  • BasicType: N (noun), S (sentence)");
    println!("  • AtomicType: Base type with adjoint level (Nʳ, Nˡ, etc.)");
    println!("  • PregroupType: Tensor product of atomic types");
    println!("  • Grammar: Lexicon + parse + reduce");
    println!("  • extract_cups: Get contractions for tensor networks");
    println!();
    println!("Key insight: Grammaticality = type reduction to S");
    println!("Reductions = cups = tensor contractions in DisCoCat");
    println!();
    println!("Next: Session 16 - DisCoCat (putting meanings on the wires)");
}
