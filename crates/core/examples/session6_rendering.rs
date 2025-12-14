//! Session 6: Rendering and Tracing
//!
//! Run with: cargo run --example session6_rendering
//!
//! This example demonstrates:
//! - ASCII rendering of diagrams
//! - DOT export for Graphviz
//! - Zero-cost tracing with const generics
//! - TraceNode hierarchy

use compositional_core::tracing::{Computation, ComputationExt, TraceNode, Traced};
use compositional_core::{CoreError, Diagram, Node, Port, Shape};
use std::time::Duration;

/// Simple operations for our diagrams
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum Op {
    Input,
    Linear,
    ReLU,
    Softmax,
    Output,
}

// ============================================================================
// Example Computations for Tracing
// ============================================================================

#[derive(Clone)]
struct FastComputation;

impl Computation for FastComputation {
    type Input = i32;
    type Output = i32;

    async fn run(&self, input: Self::Input) -> Result<Self::Output, CoreError> {
        Ok(input * 2)
    }

    fn name(&self) -> &'static str {
        "FastComputation"
    }
}

#[derive(Clone)]
struct SlowComputation {
    delay_ms: u64,
}

impl Computation for SlowComputation {
    type Input = String;
    type Output = String;

    async fn run(&self, input: Self::Input) -> Result<Self::Output, CoreError> {
        tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
        Ok(format!("Processed: {}", input))
    }

    fn name(&self) -> &'static str {
        "SlowComputation"
    }
}

#[tokio::main]
async fn main() -> Result<(), CoreError> {
    println!("=== Session 6: Rendering and Tracing ===\n");

    // -------------------------------------------------------------------------
    // Create a Sample Diagram
    // -------------------------------------------------------------------------
    println!("1. Creating a Neural Network Diagram");
    println!("-------------------------------------");

    let mut diagram: Diagram<Op> = Diagram::new();

    // Input layer
    let input = diagram.add_node(Node::new(
        Op::Input,
        vec![],
        vec![Port::new(Shape::f32_vector(784))],
    ));

    // Hidden layer 1
    let linear1 = diagram.add_node(Node::new(
        Op::Linear,
        vec![Port::new(Shape::f32_vector(784))],
        vec![Port::new(Shape::f32_vector(256))],
    ));

    // ReLU activation
    let relu = diagram.add_node(Node::new(
        Op::ReLU,
        vec![Port::new(Shape::f32_vector(256))],
        vec![Port::new(Shape::f32_vector(256))],
    ));

    // Hidden layer 2
    let linear2 = diagram.add_node(Node::new(
        Op::Linear,
        vec![Port::new(Shape::f32_vector(256))],
        vec![Port::new(Shape::f32_vector(10))],
    ));

    // Softmax output
    let softmax = diagram.add_node(Node::new(
        Op::Softmax,
        vec![Port::new(Shape::f32_vector(10))],
        vec![Port::new(Shape::f32_vector(10))],
    ));

    // Connect the layers
    diagram.connect(input, 0, linear1, 0)?;
    diagram.connect(linear1, 0, relu, 0)?;
    diagram.connect(relu, 0, linear2, 0)?;
    diagram.connect(linear2, 0, softmax, 0)?;

    // Set boundaries
    diagram.set_outputs(vec![(softmax, 0)]);

    println!(
        "Created diagram: {} nodes, {} edges\n",
        diagram.node_count(),
        diagram.edge_count()
    );

    // -------------------------------------------------------------------------
    // ASCII Rendering
    // -------------------------------------------------------------------------
    println!("2. ASCII Rendering");
    println!("------------------");
    println!("{}", diagram.render_ascii());

    // -------------------------------------------------------------------------
    // DOT Export
    // -------------------------------------------------------------------------
    println!("3. DOT Export (for Graphviz)");
    println!("----------------------------");
    println!("```dot");
    println!("{}", diagram.to_dot());
    println!("```");
    println!("Save to file and run: dot -Tsvg diagram.dot -o diagram.svg\n");

    // -------------------------------------------------------------------------
    // Tracing: Disabled (Zero Cost)
    // -------------------------------------------------------------------------
    println!("4. Tracing Disabled (Zero Cost)");
    println!("-------------------------------");

    let comp = FastComputation;
    let traced_off: Traced<FastComputation, false> = Traced::new(comp.clone());

    let result = traced_off.run(42).await?;
    println!("Input:  42");
    println!("Output: {} (no trace attached)", result);
    println!("Type:   Traced<FastComputation, false>");
    println!("Cost:   Zero overhead - wrapper compiles away\n");

    // -------------------------------------------------------------------------
    // Tracing: Enabled
    // -------------------------------------------------------------------------
    println!("5. Tracing Enabled");
    println!("------------------");

    let traced_on: Traced<FastComputation, true> = Traced::new(comp);

    let (result, trace) = traced_on.run(42).await?;
    println!("Input:  42");
    println!("Output: {}", result);
    println!("Trace:  {} took {:?}", trace.name, trace.duration);
    println!("Type:   Traced<FastComputation, true>\n");

    // -------------------------------------------------------------------------
    // Timing Accuracy
    // -------------------------------------------------------------------------
    println!("6. Timing Accuracy");
    println!("------------------");

    let slow = SlowComputation { delay_ms: 50 };
    let traced_slow: Traced<SlowComputation, true> = Traced::new(slow);

    let (result, trace) = traced_slow.run("Hello".to_string()).await?;
    println!("Result: {}", result);
    println!("Duration: {:?} (should be ~50ms)", trace.duration);
    println!();

    // -------------------------------------------------------------------------
    // Extension Trait
    // -------------------------------------------------------------------------
    println!("7. Extension Trait (.traced() / .untraced())");
    println!("--------------------------------------------");

    // Using .traced() convenience method
    let (result, trace) = FastComputation.traced().run(100).await?;
    println!("FastComputation.traced().run(100)");
    println!("  Result: {}", result);
    println!("  Trace:  {}", trace.name);

    // Using .untraced() convenience method
    let result = FastComputation.untraced().run(100).await?;
    println!("FastComputation.untraced().run(100)");
    println!("  Result: {} (no trace)", result);
    println!();

    // -------------------------------------------------------------------------
    // TraceNode Hierarchy
    // -------------------------------------------------------------------------
    println!("8. TraceNode Hierarchy");
    println!("----------------------");

    // Build a trace tree manually (in practice, this comes from composed computations)
    let child1 = TraceNode::new("WebSearch", Duration::from_millis(45));
    let child2 = TraceNode::new("DatabaseQuery", Duration::from_millis(30));
    let child3 = TraceNode::new("LLMCall", Duration::from_millis(120));

    let parent = TraceNode::new("Pipeline", Duration::from_millis(200))
        .with_child(child1)
        .with_child(child2)
        .with_child(child3);

    println!("{}", parent.display());

    // -------------------------------------------------------------------------
    // Compile-Time Selection
    // -------------------------------------------------------------------------
    println!("9. Compile-Time Selection");
    println!("-------------------------");
    println!("// In your code:");
    println!("// #[cfg(debug_assertions)]");
    println!("// pub type AutoTraced<C> = Traced<C, true>;   // Debug: tracing ON");
    println!("//");
    println!("// #[cfg(not(debug_assertions))]");
    println!("// pub type AutoTraced<C> = Traced<C, false>;  // Release: tracing OFF");
    println!();
    println!("cargo run           → Full tracing");
    println!("cargo run --release → Zero overhead");

    println!("\n=== Session 6 Complete ===");
    Ok(())
}
