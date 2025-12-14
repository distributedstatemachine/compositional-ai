//! Session 19: Operads — Multi-Input Wiring Constraints
//!
//! Run with: cargo run -p compositional-core --example session19_operads
//!
//! This example demonstrates how operads enforce arity constraints that
//! plain diagrams don't catch. While diagrams allow any wiring if types match,
//! operads ensure operations receive exactly the right number of inputs.
//!
//! # Key Insight
//!
//! Diagrams: nodes + edges, any wiring allowed if types match.
//! Operads: explicit nesting structure — enforce arity and hierarchy.

use compositional_core::operad::{op, OperadError, Operation, WiringPlan};
use compositional_core::Shape;

fn main() {
    println!("=== Session 19: Operads — Multi-Input Wiring Constraints ===\n");

    // -------------------------------------------------------------------------
    // 1. The Problem: Arity Mismatches
    // -------------------------------------------------------------------------
    println!("1. The Problem: Arity Mismatches");
    println!("--------------------------------\n");

    println!("Consider an LLM tool-use pipeline:");
    println!();
    println!("  Agent(3) — takes 3 tool outputs, produces 1 response");
    println!("    ├── Tool(\"search\")    — 1 query  → 1 result");
    println!("    ├── Tool(\"calculate\") — 2 numbers → 1 result");
    println!("    └── Tool(\"fetch\")     — 1 url    → 1 result");
    println!();
    println!("  Valid:   Agent receives (search, calc, fetch) — 3 inputs ✓");
    println!("  Invalid: Agent receives (search, calc)        — 2 inputs ✗");
    println!();
    println!("Plain diagrams can't catch this. Operads can!\n");

    // -------------------------------------------------------------------------
    // 2. Defining Operations with Arity
    // -------------------------------------------------------------------------
    println!("2. Defining Operations with Arity");
    println!("---------------------------------\n");

    // Define an agent that expects exactly 3 tool outputs
    let agent = Operation::new("ReasoningAgent", 3)
        .with_inputs(vec![
            Shape::f32_vector(256), // search result
            Shape::f32_vector(256), // calculator result
            Shape::f32_vector(256), // fetch result
        ])
        .with_outputs(vec![Shape::f32_vector(512)]); // response

    println!("Agent: {}", agent);
    println!(
        "  Arity: {} (requires exactly {} inputs)",
        agent.arity, agent.arity
    );
    println!();

    // Define tools (arity 0 = no external inputs needed)
    let search = Operation::new("SearchTool", 0).with_outputs(vec![Shape::f32_vector(256)]);

    let calculator = Operation::new("CalculatorTool", 0).with_outputs(vec![Shape::f32_vector(256)]);

    let fetch = Operation::new("WebFetchTool", 0).with_outputs(vec![Shape::f32_vector(256)]);

    println!("Tools:");
    println!("  {}", search);
    println!("  {}", calculator);
    println!("  {}", fetch);
    println!();

    // -------------------------------------------------------------------------
    // 3. Valid Wiring Plan
    // -------------------------------------------------------------------------
    println!("3. Valid Wiring Plan");
    println!("--------------------\n");

    let valid_plan = WiringPlan::new(agent.clone())
        .with_inner(vec![search.clone(), calculator.clone(), fetch.clone()])
        .with_wiring(vec![
            (0, 0), // search output → agent slot 0
            (1, 1), // calculator output → agent slot 1
            (2, 2), // fetch output → agent slot 2
        ]);

    println!("{}", valid_plan);

    match valid_plan.validate() {
        Ok(()) => println!("✓ Validation passed: All 3 slots wired correctly!\n"),
        Err(e) => println!("✗ Validation failed: {}\n", e),
    }

    // -------------------------------------------------------------------------
    // 4. Invalid: Arity Mismatch
    // -------------------------------------------------------------------------
    println!("4. Invalid: Arity Mismatch");
    println!("--------------------------\n");

    let bad_plan_arity = WiringPlan::new(agent.clone())
        .with_inner(vec![search.clone(), calculator.clone()]) // Only 2 tools!
        .with_wiring(vec![(0, 0), (1, 1)]);

    println!("Attempting to wire only 2 tools to 3-arity agent...");

    match bad_plan_arity.validate() {
        Ok(()) => println!("✓ Validation passed (unexpected!)\n"),
        Err(OperadError::ArityMismatch {
            operation,
            expected,
            got,
        }) => {
            println!(
                "✗ ArityMismatch: '{}' expected {} inputs, got {}",
                operation, expected, got
            );
            println!("  This is exactly what operads catch!\n");
        }
        Err(e) => println!("✗ Other error: {}\n", e),
    }

    // -------------------------------------------------------------------------
    // 5. Invalid: Shape Mismatch
    // -------------------------------------------------------------------------
    println!("5. Invalid: Shape Mismatch");
    println!("--------------------------\n");

    let wrong_shape_tool =
        Operation::new("WrongShapeTool", 0).with_outputs(vec![Shape::f32_vector(128)]); // Wrong size!

    let bad_plan_shape = WiringPlan::new(agent.clone())
        .with_inner(vec![search.clone(), wrong_shape_tool, fetch.clone()])
        .with_wiring(vec![(0, 0), (1, 1), (2, 2)]);

    println!("Attempting to wire tool with wrong output shape...");
    println!("  Agent expects f32[256] at slot 1");
    println!("  Tool produces f32[128]");

    match bad_plan_shape.validate() {
        Ok(()) => println!("✓ Validation passed (unexpected!)\n"),
        Err(OperadError::ShapeMismatch {
            slot,
            expected,
            got,
        }) => {
            println!(
                "✗ ShapeMismatch at slot {}: expected {}, got {}\n",
                slot, expected, got
            );
        }
        Err(e) => println!("✗ Other error: {}\n", e),
    }

    // -------------------------------------------------------------------------
    // 6. Invalid: Duplicate Wiring
    // -------------------------------------------------------------------------
    println!("6. Invalid: Duplicate Wiring");
    println!("----------------------------\n");

    let bad_plan_dup = WiringPlan::new(agent.clone())
        .with_inner(vec![search.clone(), calculator.clone(), fetch.clone()])
        .with_wiring(vec![
            (0, 0),
            (1, 0), // Wiring to slot 0 again!
            (2, 2),
        ]);

    println!("Attempting to wire two tools to the same slot...");

    match bad_plan_dup.validate() {
        Ok(()) => println!("✓ Validation passed (unexpected!)\n"),
        Err(OperadError::DuplicateWiring { slot }) => {
            println!("✗ DuplicateWiring: slot {} is wired multiple times\n", slot);
        }
        Err(e) => println!("✗ Other error: {}\n", e),
    }

    // -------------------------------------------------------------------------
    // 7. Invalid: Unwired Slot
    // -------------------------------------------------------------------------
    println!("7. Invalid: Unwired Slot");
    println!("------------------------\n");

    let bad_plan_unwired = WiringPlan::new(agent.clone())
        .with_inner(vec![search.clone(), calculator.clone(), fetch.clone()])
        .with_wiring(vec![
            (0, 0),
            (1, 2), // Skip slot 1!
            (2, 2), // This would be duplicate anyway
        ]);

    println!("Attempting to leave slot 1 unwired...");

    match bad_plan_unwired.validate() {
        Ok(()) => println!("✓ Validation passed (unexpected!)\n"),
        Err(e) => println!("✗ Error: {}\n", e),
    }

    // -------------------------------------------------------------------------
    // 8. Auto-Wiring for Simple Cases
    // -------------------------------------------------------------------------
    println!("8. Auto-Wiring for Simple Cases");
    println!("-------------------------------\n");

    let auto_plan = WiringPlan::new(agent.clone())
        .with_inner(vec![search.clone(), calculator.clone(), fetch.clone()])
        .auto_wire(); // Automatically wire: inner[i] → slot[i]

    println!("Using auto_wire() for simple 1:1 wiring:");
    println!("  Wiring: {:?}", auto_plan.wiring);

    match auto_plan.validate() {
        Ok(()) => println!("✓ Validation passed!\n"),
        Err(e) => println!("✗ Error: {}\n", e),
    }

    // -------------------------------------------------------------------------
    // 9. Converting to Diagram
    // -------------------------------------------------------------------------
    println!("9. Converting to Diagram");
    println!("------------------------\n");

    println!("Operads can be converted to Diagrams (loses arity enforcement):");

    let diagram = auto_plan.to_diagram().unwrap();
    println!("  Nodes: {}", diagram.node_count());
    println!("  Edges: {}", diagram.edge_count());
    println!();

    // -------------------------------------------------------------------------
    // 10. Multi-Agent Orchestration
    // -------------------------------------------------------------------------
    println!("10. Multi-Agent Orchestration");
    println!("-----------------------------\n");

    let orchestrator = Operation::new("Orchestrator", 4)
        .with_inputs(vec![
            Shape::f32_vector(256),
            Shape::f32_vector(256),
            Shape::f32_vector(256),
            Shape::f32_vector(256),
        ])
        .with_outputs(vec![Shape::f32_vector(1024)]);

    let researcher = Operation::new("ResearchAgent", 0).with_outputs(vec![Shape::f32_vector(256)]);
    let analyst = Operation::new("AnalystAgent", 0).with_outputs(vec![Shape::f32_vector(256)]);
    let writer = Operation::new("WriterAgent", 0).with_outputs(vec![Shape::f32_vector(256)]);
    let reviewer = Operation::new("ReviewerAgent", 0).with_outputs(vec![Shape::f32_vector(256)]);

    let multi_agent_plan = WiringPlan::new(orchestrator)
        .with_inner(vec![researcher, analyst, writer, reviewer])
        .auto_wire();

    println!("{}", multi_agent_plan);

    match multi_agent_plan.validate() {
        Ok(()) => println!("✓ Multi-agent plan validated!\n"),
        Err(e) => println!("✗ Error: {}\n", e),
    }

    // -------------------------------------------------------------------------
    // 11. Using the Builder Pattern
    // -------------------------------------------------------------------------
    println!("11. Using the Builder Pattern");
    println!("-----------------------------\n");

    let my_op = op("MyCustomAgent")
        .input(Shape::f32_vector(100))
        .input(Shape::f32_vector(200))
        .output(Shape::f32_vector(300))
        .build();

    println!("Built with builder: {}", my_op);
    println!("  Arity: {}", my_op.arity);
    println!();

    // -------------------------------------------------------------------------
    // 12. What Operads Catch That Diagrams Don't
    // -------------------------------------------------------------------------
    println!("12. What Operads Catch That Diagrams Don't");
    println!("------------------------------------------\n");

    println!("  | Check              | Diagram | Operad |");
    println!("  |--------------------|---------|--------|");
    println!("  | Shape compatibility| ✓       | ✓      |");
    println!("  | Arity (input count)| ✗       | ✓      |");
    println!("  | Duplicate wiring   | ✗       | ✓      |");
    println!("  | Unwired slots      | ✗       | ✓      |");
    println!("  | Hierarchical nest  | ✗       | ✓      |");
    println!();

    // -------------------------------------------------------------------------
    // Exercises
    // -------------------------------------------------------------------------
    println!("=== Exercises ===\n");

    println!("1. Create a 5-tool pipeline and validate it.");
    println!("2. Try creating circular wiring (what happens?).");
    println!("3. Create nested operations: tool outputs feed into intermediate");
    println!("   combiners, which feed into the final agent.");
    println!("4. Convert a valid WiringPlan to Diagram and render it.");
    println!();

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    println!("=== Session 19 Complete ===\n");

    println!("Key Takeaways:");
    println!("  1. Operations have explicit arity — fixed number of inputs");
    println!("  2. WiringPlan validates: arity, shapes, no duplicates, all wired");
    println!("  3. Operads catch composition errors that diagrams miss");
    println!("  4. Use auto_wire() for simple 1:1 mappings");
    println!("  5. Convert to Diagram when you need graph operations");
    println!();

    println!("Operads = Diagrams + Arity Enforcement!");
    println!();

    println!("Next: Session 20 — Integration and polish for capstone");
}
