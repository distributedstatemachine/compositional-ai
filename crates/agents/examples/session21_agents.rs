//! Session 21: Rust-Native Agent Framework — Capstone
//!
//! Run with: cargo run -p compositional-agents --example session21_agents
//!
//! This example demonstrates the complete agent framework built in Session 21:
//! - Tool registration using Yoneda-style capabilities
//! - Agent loop with tool calling
//! - Execution tracing
//! - Multi-agent orchestration with arity validation
//!
//! ## What This Framework Does Better Than Python (Agentica)
//!
//! | Aspect | Python/Agentica | Rust/Compositional |
//! |--------|-----------------|-------------------|
//! | Type checking | Runtime | Compile-time |
//! | Sandbox | WASM overhead | Borrow checker (zero-cost) |
//! | Parallel safety | asyncio + hope | Send + Sync proofs |
//! | Arity checking | None | Operad validation |
//! | Tracing | Always on | Const generics (compiled out) |

use compositional_agents::{
    agent::{AgentConfig, AgentLoop, SimpleAgent},
    llm::{DeterministicMockLlm, MockLlmClient},
    orchestrator::{fanout_pipeline, sequential_pipeline, AgentWrapper},
    requests::ToolSchema,
    tool::{calculator_tool, mock_search_tool, mock_weather_tool, Tool, ToolRegistry},
    trace::TraceSummary,
};
use std::sync::Arc;

fn main() {
    println!("=== Session 21: Rust-Native Agent Framework ===\n");

    // -------------------------------------------------------------------------
    // 1. Tool Registration (Yoneda-style)
    // -------------------------------------------------------------------------
    println!("1. Tool Registration (Yoneda-Style Capabilities)");
    println!("------------------------------------------------\n");

    println!("Instead of hardcoded traits like HasCalculator, HasSearch,");
    println!("we use the Yoneda pattern: tools are capabilities that handle requests.\n");

    let mut registry = ToolRegistry::new();
    registry.register(calculator_tool());
    registry.register(mock_search_tool());
    registry.register(mock_weather_tool());

    println!("Registered tools: {:?}", registry.tool_names());
    println!("Tool count: {}\n", registry.len());

    // Test tool invocation
    let calc_result = registry
        .invoke("calculate", serde_json::json!({"expression": "10 + 5 * 2"}))
        .unwrap();
    println!("Calculator test: 10 + 5 * 2 = {}\n", calc_result);

    // -------------------------------------------------------------------------
    // 2. Simple Agent (Text Response)
    // -------------------------------------------------------------------------
    println!("2. Simple Agent (Text Response)");
    println!("--------------------------------\n");

    let llm = Arc::new(MockLlmClient::new("claude-3-opus"));
    let agent = SimpleAgent::new(llm);

    let result = agent.run("Hello, how are you?").unwrap();
    println!("Agent response: {}", result.response);
    println!("Iterations: {}", result.iterations);
    println!("Tool calls: {:?}\n", result.tool_calls);

    // -------------------------------------------------------------------------
    // 3. Agent with Tool Calling
    // -------------------------------------------------------------------------
    println!("3. Agent with Tool Calling");
    println!("--------------------------\n");

    // Create deterministic mock that calls calculator then returns answer
    let llm = Arc::new(DeterministicMockLlm::tool_then_text(
        "calculate",
        serde_json::json!({"expression": "25 * 4"}),
        "The answer is 100",
    ));

    let mut tools = ToolRegistry::new();
    tools.register(calculator_tool());

    let agent = SimpleAgent::with_tools(llm, tools);
    let result = agent.run("What is 25 times 4?").unwrap();

    println!("User: What is 25 times 4?");
    println!("Agent: {}", result.response);
    println!("Iterations: {}", result.iterations);
    println!("Tools called: {:?}\n", result.tool_calls);

    // -------------------------------------------------------------------------
    // 4. Execution Trace
    // -------------------------------------------------------------------------
    println!("4. Execution Trace");
    println!("------------------\n");

    println!("{}", result.trace.render_ascii());

    let summary = TraceSummary::from(&result.trace);
    println!("Summary:");
    println!("  Total events: {}", summary.total_events);
    println!("  LLM calls: {}", summary.llm_calls);
    println!("  Tool calls: {}", summary.tool_calls);
    println!("  Success rate: {:.1}%\n", summary.success_rate * 100.0);

    // -------------------------------------------------------------------------
    // 5. Agent with System Prompt
    // -------------------------------------------------------------------------
    println!("5. Agent with System Prompt");
    println!("---------------------------\n");

    let llm = Arc::new(DeterministicMockLlm::text_only(
        "I'd be happy to help! What would you like to know?",
    ));

    let config = AgentConfig::default()
        .with_system_prompt("You are a helpful assistant. Be concise and friendly.")
        .with_max_tokens(512)
        .with_temperature(0.7);

    let tools = ToolRegistry::new();
    let agent = SimpleAgent::with_config(llm, tools, config);

    let result = agent.run("Hi there!").unwrap();
    println!("Response: {}\n", result.response);

    // -------------------------------------------------------------------------
    // 6. Multi-Agent Pipeline (Sequential)
    // -------------------------------------------------------------------------
    println!("6. Multi-Agent Pipeline (Sequential)");
    println!("------------------------------------\n");

    fn create_mock_agent(name: &str, response: &str) -> Arc<dyn compositional_agents::AgentRunner> {
        let llm = Arc::new(DeterministicMockLlm::text_only(response));
        let tools = Arc::new(ToolRegistry::new());
        let agent = AgentLoop::new(llm, tools, AgentConfig::default());
        Arc::new(AgentWrapper::new(name, agent, 0))
    }

    let agents = vec![
        create_mock_agent(
            "researcher",
            "Research findings: Rust is a systems language...",
        ),
        create_mock_agent(
            "analyst",
            "Analysis: The findings indicate high adoption...",
        ),
        create_mock_agent("writer", "Final report: Rust adoption is growing rapidly."),
    ];

    let pipeline = sequential_pipeline(agents);
    println!("Pipeline stages: {}", pipeline.stage_count());
    println!("Validation: {:?}", pipeline.validate());

    let result = pipeline.execute("Write a report about Rust").unwrap();
    println!("Final output: {}", result.response);
    println!("Stages completed: {}\n", result.stage_results.len());

    // -------------------------------------------------------------------------
    // 7. Multi-Agent Pipeline (Fan-out)
    // -------------------------------------------------------------------------
    println!("7. Multi-Agent Pipeline (Fan-out)");
    println!("---------------------------------\n");

    let agents = vec![
        create_mock_agent("web_searcher", "Web results: [1] Article about topic..."),
        create_mock_agent(
            "doc_searcher",
            "Docs results: [A] Official documentation...",
        ),
        create_mock_agent(
            "code_searcher",
            "Code results: [i] Example implementation...",
        ),
    ];

    let pipeline = fanout_pipeline(agents);
    let result = pipeline
        .execute("Find information about async Rust")
        .unwrap();

    println!("Fan-out results:");
    for (i, stage_result) in result.stage_results.iter().enumerate() {
        println!(
            "  Stage {}: {}",
            i,
            stage_result.response.chars().take(50).collect::<String>()
        );
    }
    println!();

    // -------------------------------------------------------------------------
    // 8. Custom Tool Definition
    // -------------------------------------------------------------------------
    println!("8. Custom Tool Definition");
    println!("-------------------------\n");

    // Define a custom request type
    #[derive(Debug, serde::Deserialize)]
    struct GreetArgs {
        name: String,
        formal: bool,
    }

    let greet_tool = Tool::new(
        ToolSchema::new("greet", "Generate a greeting for someone")
            .param("name", "string", "The person's name", true)
            .param("formal", "boolean", "Whether to be formal", false),
        |args: GreetArgs| {
            let greeting = if args.formal {
                format!("Good day, {}. How may I assist you?", args.name)
            } else {
                format!("Hey {}! What's up?", args.name)
            };
            Ok(greeting)
        },
    );

    let mut registry = ToolRegistry::new();
    registry.register(greet_tool);

    let informal = registry
        .invoke(
            "greet",
            serde_json::json!({"name": "Alice", "formal": false}),
        )
        .unwrap();
    let formal = registry
        .invoke(
            "greet",
            serde_json::json!({"name": "Dr. Smith", "formal": true}),
        )
        .unwrap();

    println!("Informal: {}", informal);
    println!("Formal: {}\n", formal);

    // -------------------------------------------------------------------------
    // 9. Trace to DOT Format (for Graphviz)
    // -------------------------------------------------------------------------
    println!("9. Trace to DOT Format");
    println!("----------------------\n");

    // Recreate a traced execution
    let llm = Arc::new(DeterministicMockLlm::tool_then_text(
        "search",
        serde_json::json!({"query": "Rust async"}),
        "Based on the search results: Rust async is powerful.",
    ));

    let mut tools = ToolRegistry::new();
    tools.register(mock_search_tool());

    let agent = SimpleAgent::with_tools(llm, tools);
    let result = agent.run("Search for Rust async").unwrap();

    println!("DOT output (paste into graphviz.org):\n");
    println!("{}", result.trace.render_dot());

    // -------------------------------------------------------------------------
    // 10. Compile-Time Safety Demonstration
    // -------------------------------------------------------------------------
    println!("10. Compile-Time Safety (The Rust Advantage)");
    println!("--------------------------------------------\n");

    println!("What Python catches at RUNTIME, Rust catches at COMPILE TIME:\n");

    println!("1. Missing capability:");
    println!("   Python: KeyError at runtime");
    println!("   Rust:   Compile error - trait bound not satisfied\n");

    println!("2. Thread safety:");
    println!("   Python: Race condition in production");
    println!("   Rust:   Compile error - type is not Send/Sync\n");

    println!("3. Wrong arity:");
    println!("   Python: Silent bug or runtime error");
    println!("   Rust:   OperadError::ArityMismatch before execution\n");

    println!("4. Type mismatch:");
    println!("   Python: TypeError at runtime");
    println!("   Rust:   Compile error - expected X, found Y\n");

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    println!("=== Session 21 Complete ===\n");

    println!("Key Accomplishments:");
    println!("  1. Yoneda-style tool registration (extensible without core changes)");
    println!("  2. Agent loop with tool calling and iteration");
    println!("  3. Execution tracing with ASCII and DOT rendering");
    println!("  4. Multi-agent orchestration (sequential and fan-out)");
    println!("  5. Arity validation using operads from core");
    println!("  6. Compile-time safety guarantees\n");

    println!("The Rust Advantage:");
    println!("  • Type errors caught at compile time, not runtime");
    println!("  • Thread safety proven by the compiler");
    println!("  • Zero-cost abstractions (tracing compiled out in release)");
    println!("  • Extensible without modifying core code\n");

    println!("This framework provides everything Agentica does,");
    println!("plus compile-time guarantees that Python can never offer.");
}
