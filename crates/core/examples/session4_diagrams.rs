//! Session 4: Diagrams - Programs as Pictures
//!
//! Run with: cargo run --example session4_diagrams
//!
//! This example demonstrates:
//! - Creating diagrams (boxes + wires)
//! - Adding nodes with typed ports
//! - Connecting nodes (shape-checked wiring)
//! - Sequential composition (f ; g)
//! - Parallel composition (f ⊗ g)

use compositional_core::{CoreError, Diagram, Node, Port, Shape};

/// Simple operations for our example diagrams
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum Op {
    Input,
    Linear { in_dim: usize, out_dim: usize },
    ReLU,
    Add,
    Output,
}

fn main() {
    println!("=== Session 4: Diagrams ===\n");

    // -------------------------------------------------------------------------
    // Creating a Simple Diagram
    // -------------------------------------------------------------------------
    println!("1. Creating a Simple Diagram");
    println!("-----------------------------");

    let mut diagram: Diagram<Op> = Diagram::new();

    // Add an input node (no inputs, one output)
    let input = diagram.add_node(Node::new(
        Op::Input,
        vec![],                                  // no input ports
        vec![Port::new(Shape::f32_vector(784))], // output: 784-dim vector
    ));

    // Add a linear layer (one input, one output)
    let linear = diagram.add_node(Node::new(
        Op::Linear {
            in_dim: 784,
            out_dim: 128,
        },
        vec![Port::new(Shape::f32_vector(784))], // input: 784-dim
        vec![Port::new(Shape::f32_vector(128))], // output: 128-dim
    ));

    // Add ReLU activation
    let relu = diagram.add_node(Node::new(
        Op::ReLU,
        vec![Port::new(Shape::f32_vector(128))],
        vec![Port::new(Shape::f32_vector(128))],
    ));

    println!("Created diagram with {} nodes", diagram.node_count());
    println!();

    // -------------------------------------------------------------------------
    // Connecting Nodes
    // -------------------------------------------------------------------------
    println!("2. Connecting Nodes");
    println!("-------------------");

    // Connect input -> linear (shapes match: 784 -> 784)
    diagram
        .connect(input, 0, linear, 0)
        .expect("Should connect");
    println!("Connected Input -> Linear");

    // Connect linear -> relu (shapes match: 128 -> 128)
    diagram.connect(linear, 0, relu, 0).expect("Should connect");
    println!("Connected Linear -> ReLU");

    println!("Diagram now has {} edges", diagram.edge_count());
    println!();

    // -------------------------------------------------------------------------
    // Shape Mismatch Detection
    // -------------------------------------------------------------------------
    println!("3. Shape Mismatch Detection");
    println!("---------------------------");

    let mut bad_diagram: Diagram<Op> = Diagram::new();

    let producer = bad_diagram.add_node(Node::new(
        Op::Linear {
            in_dim: 10,
            out_dim: 20,
        },
        vec![Port::new(Shape::f32_vector(10))],
        vec![Port::new(Shape::f32_vector(20))], // outputs 20-dim
    ));

    let consumer = bad_diagram.add_node(Node::new(
        Op::Linear {
            in_dim: 50,
            out_dim: 10,
        },
        vec![Port::new(Shape::f32_vector(50))], // expects 50-dim!
        vec![Port::new(Shape::f32_vector(10))],
    ));

    match bad_diagram.connect(producer, 0, consumer, 0) {
        Ok(_) => println!("Connected (unexpected!)"),
        Err(CoreError::ShapeMismatch { expected, got }) => {
            println!("Shape mismatch detected!");
            println!("  Expected: {}", expected);
            println!("  Got:      {}", got);
        }
        Err(e) => println!("Other error: {}", e),
    }
    println!();

    // -------------------------------------------------------------------------
    // Sequential Composition: f ; g
    // -------------------------------------------------------------------------
    println!("4. Sequential Composition (f ; g)");
    println!("---------------------------------");

    // Create f: vector(10) -> vector(20)
    let f = Diagram::from_node(Node::new(
        Op::Linear {
            in_dim: 10,
            out_dim: 20,
        },
        vec![Port::new(Shape::f32_vector(10))],
        vec![Port::new(Shape::f32_vector(20))],
    ));

    // Create g: vector(20) -> vector(30)
    let g = Diagram::from_node(Node::new(
        Op::Linear {
            in_dim: 20,
            out_dim: 30,
        },
        vec![Port::new(Shape::f32_vector(20))],
        vec![Port::new(Shape::f32_vector(30))],
    ));

    println!("f: {:?} -> {:?}", f.input_shapes(), f.output_shapes());
    println!("g: {:?} -> {:?}", g.input_shapes(), g.output_shapes());

    // Compose: f ; g
    let fg = f.then(g).expect("Shapes should match");

    println!("f ; g: {:?} -> {:?}", fg.input_shapes(), fg.output_shapes());
    println!(
        "f ; g has {} nodes, {} edges",
        fg.node_count(),
        fg.edge_count()
    );
    println!();

    // -------------------------------------------------------------------------
    // Parallel Composition: f ⊗ g
    // -------------------------------------------------------------------------
    println!("5. Parallel Composition (f ⊗ g)");
    println!("-------------------------------");

    // Create f: vector(10) -> vector(20)
    let f = Diagram::from_node(Node::new(
        Op::Linear {
            in_dim: 10,
            out_dim: 20,
        },
        vec![Port::new(Shape::f32_vector(10))],
        vec![Port::new(Shape::f32_vector(20))],
    ));

    // Create g: vector(30) -> vector(40)
    let g = Diagram::from_node(Node::new(
        Op::Linear {
            in_dim: 30,
            out_dim: 40,
        },
        vec![Port::new(Shape::f32_vector(30))],
        vec![Port::new(Shape::f32_vector(40))],
    ));

    println!("f: {:?} -> {:?}", f.input_shapes(), f.output_shapes());
    println!("g: {:?} -> {:?}", g.input_shapes(), g.output_shapes());

    // Parallel: f ⊗ g
    let fg_parallel = f.tensor(g);

    println!(
        "f ⊗ g: {:?} -> {:?}",
        fg_parallel.input_shapes(),
        fg_parallel.output_shapes()
    );
    println!(
        "f ⊗ g has {} nodes, {} edges (no connection between parallel)",
        fg_parallel.node_count(),
        fg_parallel.edge_count()
    );
    println!();

    // -------------------------------------------------------------------------
    // Interchange Law
    // -------------------------------------------------------------------------
    println!("6. Interchange Law");
    println!("------------------");
    println!("(f₁ ; g₁) ⊗ (f₂ ; g₂) = (f₁ ⊗ f₂) ; (g₁ ⊗ g₂)");
    println!();

    // Create four morphisms
    let f1 = Diagram::from_node(Node::new(
        Op::Linear {
            in_dim: 10,
            out_dim: 20,
        },
        vec![Port::new(Shape::f32_vector(10))],
        vec![Port::new(Shape::f32_vector(20))],
    ));
    let g1 = Diagram::from_node(Node::new(
        Op::Linear {
            in_dim: 20,
            out_dim: 30,
        },
        vec![Port::new(Shape::f32_vector(20))],
        vec![Port::new(Shape::f32_vector(30))],
    ));
    let f2 = Diagram::from_node(Node::new(
        Op::Linear {
            in_dim: 40,
            out_dim: 50,
        },
        vec![Port::new(Shape::f32_vector(40))],
        vec![Port::new(Shape::f32_vector(50))],
    ));
    let g2 = Diagram::from_node(Node::new(
        Op::Linear {
            in_dim: 50,
            out_dim: 60,
        },
        vec![Port::new(Shape::f32_vector(50))],
        vec![Port::new(Shape::f32_vector(60))],
    ));

    // Left side: (f₁ ; g₁) ⊗ (f₂ ; g₂)
    let f1g1 = f1.clone().then(g1.clone()).unwrap();
    let f2g2 = f2.clone().then(g2.clone()).unwrap();
    let left = f1g1.tensor(f2g2);

    // Right side: (f₁ ⊗ f₂) ; (g₁ ⊗ g₂)
    let f1_f2 = f1.tensor(f2);
    let g1_g2 = g1.tensor(g2);
    let right = f1_f2.then(g1_g2).unwrap();

    println!(
        "Left:  inputs={:?}, outputs={:?}",
        left.input_shapes(),
        left.output_shapes()
    );
    println!(
        "Right: inputs={:?}, outputs={:?}",
        right.input_shapes(),
        right.output_shapes()
    );
    println!(
        "Same boundaries? {}",
        left.input_shapes() == right.input_shapes()
            && left.output_shapes() == right.output_shapes()
    );

    println!("\n=== Session 4 Complete ===");
}
