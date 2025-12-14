//! Session 3: Categories - Objects, Morphisms, and Composition
//!
//! Run with: cargo run --example session3_categories
//!
//! This example demonstrates:
//! - Finite categories with objects and morphisms
//! - Identity morphisms and composition
//! - Opposite categories (reversing arrows)
//! - Coproducts (categorical sums)
//! - Scopes for capability discovery

use compositional_core::cat::{FiniteCategory, OppositeCategory};
use compositional_core::{Coproduct, Scope};

fn main() {
    println!("=== Session 3: Categories ===\n");

    // -------------------------------------------------------------------------
    // Finite Categories
    // -------------------------------------------------------------------------
    println!("1. Finite Categories");
    println!("--------------------");

    // Create a category with objects and morphisms
    let mut cat: FiniteCategory<&str> = FiniteCategory::new();

    // Add objects (think: types)
    cat.add_object("A")
        .add_object("B")
        .add_object("C")
        .add_object("D");

    // Add morphisms (think: functions)
    // f: A → B
    // g: B → C
    // h: C → D
    cat.add_morphism("f", "A", "B")
        .add_morphism("g", "B", "C")
        .add_morphism("h", "C", "D");

    println!("Objects: {:?}", cat.objects);
    println!("Morphisms: {} total", cat.morphisms.len());
    println!("  - 4 identity morphisms (id_A, id_B, id_C, id_D)");
    println!("  - 3 added morphisms (f, g, h)");
    println!();

    // -------------------------------------------------------------------------
    // Composition
    // -------------------------------------------------------------------------
    println!("2. Composition");
    println!("--------------");

    // Compose f: A → B with g: B → C to get f;g: A → C
    if let Some(fg) = cat.compose("f", "g") {
        println!("f ; g = {} : {} → {}", fg.name, fg.dom, fg.cod);
    }

    // Identity composition: id_A ; f = f
    if let Some(result) = cat.compose("id_A", "f") {
        println!("id_A ; f = {} (identity law)", result.name);
    }

    // f ; id_B = f
    if let Some(result) = cat.compose("f", "id_B") {
        println!("f ; id_B = {} (identity law)", result.name);
    }

    // Can't compose f: A → B with h: C → D (domain mismatch)
    match cat.compose("f", "h") {
        Some(_) => println!("f ; h = composed (unexpected!)"),
        None => println!("f ; h = None (B ≠ C, can't compose)"),
    }
    println!();

    // -------------------------------------------------------------------------
    // Opposite Category
    // -------------------------------------------------------------------------
    println!("3. Opposite Category (C^op)");
    println!("---------------------------");

    // Create C^op where all arrows are reversed
    let op = OppositeCategory::new(cat);

    // In C: f: A → B
    // In C^op: f: B → A
    if let Some(f_op) = op.get_morphism("f") {
        println!("In C:    f: A → B");
        println!("In C^op: f: {} → {}", f_op.dom, f_op.cod);
    }

    // In C^op, composition order is reversed
    // (f ; g) in C becomes (g ; f) in C^op
    println!("Composition in C^op reverses order");
    println!();

    // -------------------------------------------------------------------------
    // Coproducts (Categorical Sums)
    // -------------------------------------------------------------------------
    println!("4. Coproducts");
    println!("-------------");

    // A coproduct A + B has:
    // - inl: A → A + B (left injection)
    // - inr: B → A + B (right injection)
    let coprod: Coproduct<i32, String> = Coproduct::new("Int + String", "inl", "inr");

    println!("Coproduct: {}", coprod.sum);
    println!("  Left injection:  {}", coprod.inj_left);
    println!("  Right injection: {}", coprod.inj_right);
    println!();

    // In Rust, this is like Either/Result:
    // enum Either<A, B> { Left(A), Right(B) }
    println!("Rust analogy: enum Either<A, B> {{ Left(A), Right(B) }}");
    println!();

    // -------------------------------------------------------------------------
    // Scopes (Capability Discovery)
    // -------------------------------------------------------------------------
    println!("5. Scopes");
    println!("---------");

    // Scopes map capability names to type information
    use compositional_core::TypeId;

    let mut scope1 = Scope::new();
    scope1.insert("read", TypeId("String"));
    scope1.insert("write", TypeId("Vec<u8>"));

    let mut scope2 = Scope::new();
    scope2.insert("execute", TypeId("fn()"));
    scope2.insert("write", TypeId("String")); // Overlaps with scope1

    println!("Scope 1 methods: {:?}", scope1.available_methods());
    println!("Scope 2 methods: {:?}", scope2.available_methods());

    // Merge scopes (right-biased on conflict)
    let merged = scope1.merge(&scope2);
    println!("Merged methods:  {:?}", merged.available_methods());
    println!("(On conflict, right scope wins)");
    println!();

    // -------------------------------------------------------------------------
    // Why Categories Matter
    // -------------------------------------------------------------------------
    println!("6. Why Categories Matter");
    println!("------------------------");

    println!("Categories give us:");
    println!("  - Composition laws that guarantee consistency");
    println!("  - Identity morphisms (do-nothing operations)");
    println!("  - A framework for reasoning about transformations");
    println!();
    println!("In our system:");
    println!("  - Objects = Shapes (tensor types)");
    println!("  - Morphisms = Diagram nodes (operations)");
    println!("  - Composition = Wiring diagrams together");

    println!("\n=== Session 3 Complete ===");
}
