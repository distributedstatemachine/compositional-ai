//! Session 18.5: Compile-Time Agent Safety — Lifetimes as Sandboxes
//!
//! Run with: cargo run -p compositional-games --example session18_5_scoped_agents
//!
//! This example demonstrates how Rust's type system provides **compile-time safety
//! guarantees** for agents that other languages achieve through runtime sandboxing.
//!
//! # Key Insights
//!
//! 1. **Lifetimes are sandboxes**: Agents cannot access outside their scope
//! 2. **Send/Sync are proofs**: Compiler verifies thread safety
//! 3. **Traits are capabilities**: Bound what agents can access
//! 4. **Zero cost**: All checks at compile time
//!
//! # Why Python Needs Sandboxes, Rust Doesn't
//!
//! Python agent frameworks use WASM sandboxes, MicroVMs, and runtime checks.
//! Rust provides the same guarantees at **compile time** with zero overhead.

use compositional_games::scoped::{
    parallel_execute, security_demo, AgentPipeline, DatabaseOnlyScope, FileSystemScope, FullScope,
    MockDatabase, MockFileSystem, MockLLM, OwnedAgent, ReadOnlyScope, ScopedAgent, ThreadSafeScope,
};
use std::sync::Arc;

fn main() {
    println!("=== Session 18.5: Compile-Time Agent Safety ===\n");

    // -------------------------------------------------------------------------
    // 1. The Borrow Checker as a Sandbox
    // -------------------------------------------------------------------------
    println!("1. The Borrow Checker as a Sandbox");
    println!("-----------------------------------\n");

    println!("In Python, you'd write:");
    println!("  with Sandbox() as sandbox:");
    println!("      agent = Agent(sandbox)");
    println!("      result = agent.run(task)  # Checked at runtime");
    println!("  # Hope nothing leaked...\n");

    println!("In Rust, the compiler enforces isolation:");
    println!("  {{");
    println!("      let agent = Agent::new(&resources, config);");
    println!("      agent.call(\"task\");");
    println!("  }}  // agent dropped, borrow ends");
    println!("  // resources still valid — COMPILER PROVED IT\n");

    // Demonstrate this in practice
    let scope = FullScope::new(MockDatabase::new(), MockLLM::new("gpt-4"));

    println!("Creating scope with database and LLM...");
    {
        let agent = ScopedAgent::new(&scope, "scoped-worker");
        println!("  Created agent '{}' (borrows scope)", agent.name());

        let result = agent.execute("query users").unwrap();
        println!("  Agent executed: {}", &result[..result.len().min(80)]);
        println!("  Executions: {}", agent.execution_count());
    } // agent dropped here

    // Scope is still valid!
    let data = scope.db.query("users");
    println!("  After agent dropped, scope still valid: {:?}\n", data);

    // -------------------------------------------------------------------------
    // 2. Capability-Based Security via Traits
    // -------------------------------------------------------------------------
    println!("2. Capability-Based Security via Traits");
    println!("---------------------------------------\n");

    println!("Traits define what an agent can access:");
    println!("  • HasDatabase — can query database");
    println!("  • HasLLM — can generate completions");
    println!("  • HasFileSystem — can read/write files");
    println!("  • HasReadOnlyDatabase — read-only database access\n");

    // Full access scope
    println!("FullScope: HasDatabase + HasLLM");
    let full_scope = FullScope::new(MockDatabase::new(), MockLLM::new("claude"));
    let full_agent = ScopedAgent::new(&full_scope, "full-access");
    let result = full_agent.execute("query products").unwrap();
    println!("  Result: {}\n", &result[..result.len().min(60)]);

    // Database-only scope
    println!("DatabaseOnlyScope: HasDatabase only");
    let db_scope = DatabaseOnlyScope::new(MockDatabase::new());
    let db_agent = ScopedAgent::new(&db_scope, "db-only");
    let result = db_agent.query_only("products").unwrap();
    println!("  Result: {:?}", result);
    println!("  Note: db_agent.execute() would NOT compile — no HasLLM!\n");

    // Read-only scope
    println!("ReadOnlyScope: HasReadOnlyDatabase only");
    let ro_scope = ReadOnlyScope::new(MockDatabase::new());
    let ro_agent = ScopedAgent::new(&ro_scope, "read-only");
    let result = ro_agent.read_only_query("users").unwrap();
    println!("  Result: {:?}", result);
    println!("  This agent can ONLY read — cannot write or use LLM!\n");

    // -------------------------------------------------------------------------
    // 3. Principle of Least Privilege
    // -------------------------------------------------------------------------
    println!("3. Principle of Least Privilege");
    println!("-------------------------------\n");

    println!("Different scopes provide different capabilities:\n");

    println!("  Scope Type          | Database | LLM | FileSystem | Read-Only");
    println!("  --------------------|----------|-----|------------|----------");
    println!("  FullScope           |    ✓     |  ✓  |            |");
    println!("  DatabaseOnlyScope   |    ✓     |     |            |    ✓");
    println!("  ReadOnlyScope       |          |     |            |    ✓");
    println!("  FileSystemScope     |          |     |     ✓      |");
    println!();

    println!("The compiler enforces these restrictions:");
    println!("  • If you try to use LLM on DatabaseOnlyScope → compile error");
    println!("  • If you try to write with ReadOnlyScope → compile error");
    println!("  • No runtime checks needed — all verified at compile time!\n");

    // -------------------------------------------------------------------------
    // 4. File System Access
    // -------------------------------------------------------------------------
    println!("4. File System Access");
    println!("---------------------\n");

    let fs_scope = FileSystemScope::new(MockFileSystem::new());
    let fs_agent = ScopedAgent::new(&fs_scope, "fs-worker");

    // Read existing file
    let content = fs_agent.read_file("/etc/config").unwrap();
    println!("  Read /etc/config: {}", content);

    // Write new file
    fs_agent
        .write_file("/tmp/output.txt", "Hello from agent!")
        .unwrap();
    println!("  Wrote /tmp/output.txt");

    // Read it back
    let content = fs_agent.read_file("/tmp/output.txt").unwrap();
    println!("  Read back: {}\n", content);

    // -------------------------------------------------------------------------
    // 5. Send + Sync: Parallel Safety
    // -------------------------------------------------------------------------
    println!("5. Send + Sync: Parallel Safety");
    println!("-------------------------------\n");

    println!("Rust has two marker traits for thread safety:");
    println!("  • Send: Type can be TRANSFERRED to another thread");
    println!("  • Sync: Type can be SHARED between threads via &T\n");

    println!("ThreadSafeScope is Send + Sync — compiler proves safety!");

    let shared_scope = Arc::new(ThreadSafeScope::new(
        MockDatabase::new(),
        MockLLM::new("parallel-model"),
    ));

    let tasks = vec![
        "task-alpha".to_string(),
        "task-beta".to_string(),
        "task-gamma".to_string(),
    ];

    println!("  Executing {} tasks in parallel...", tasks.len());
    let results = parallel_execute(shared_scope, tasks);

    for (i, result) in results.iter().enumerate() {
        match result {
            Ok(output) => println!("  Task {}: {}", i + 1, &output[..output.len().min(50)]),
            Err(e) => println!("  Task {}: Error: {}", i + 1, e),
        }
    }
    println!();

    // -------------------------------------------------------------------------
    // 6. What Won't Compile
    // -------------------------------------------------------------------------
    println!("6. What Won't Compile (Type System Prevents Bugs)");
    println!("-------------------------------------------------\n");

    println!("These would be COMPILE ERRORS:\n");

    println!("  // 1. Rc is not Send — can't share across threads");
    println!("  let shared = Rc::new(resources);");
    println!("  thread::spawn(move || use_resources(&shared));  // ERROR!\n");

    println!("  // 2. RefCell is not Sync — can't share mutably");
    println!("  let shared = Arc::new(RefCell::new(data));");
    println!("  thread::spawn(|| shared.borrow_mut());  // ERROR!\n");

    println!("  // 3. Agent can't outlive scope");
    println!("  let agent;");
    println!("  {{");
    println!("      let scope = Scope::new();");
    println!("      agent = Agent::new(&scope);");
    println!("  }}  // scope dropped");
    println!("  agent.execute();  // ERROR: scope doesn't live long enough\n");

    println!("  // 4. Wrong capabilities");
    println!("  let scope = ReadOnlyScope::new(...);");
    println!("  let agent = Agent::new(&scope);");
    println!("  agent.execute();  // ERROR: ReadOnlyScope doesn't impl HasLLM\n");

    // -------------------------------------------------------------------------
    // 7. Owned Agents
    // -------------------------------------------------------------------------
    println!("7. Owned Agents (When You Need Ownership)");
    println!("-----------------------------------------\n");

    let scope = FullScope::new(
        MockDatabase::new().with_data("special", "owned agent data"),
        MockLLM::new("owned-model"),
    );

    let agent = OwnedAgent::new(scope, "owned-worker");
    println!("  Created OwnedAgent '{}'", agent.name());

    let result = agent.execute("query special").unwrap();
    println!("  Result: {}", &result[..result.len().min(60)]);

    // Can recover the scope
    let recovered_scope = agent.into_scope();
    println!(
        "  Recovered scope, DB data: {:?}\n",
        recovered_scope.db.query("special")
    );

    // -------------------------------------------------------------------------
    // 8. Agent Pipelines
    // -------------------------------------------------------------------------
    println!("8. Agent Pipelines (Compositional Design)");
    println!("-----------------------------------------\n");

    let scope = FullScope::new(MockDatabase::new(), MockLLM::new("pipeline-model"));

    let pipeline = AgentPipeline::new(&scope)
        .add_stage("preprocessor")
        .add_stage("analyzer")
        .add_stage("synthesizer")
        .add_stage("formatter");

    println!("  Pipeline stages: preprocessor → analyzer → synthesizer → formatter\n");

    let results = pipeline.execute("initial query about users").unwrap();

    for result in &results {
        println!(
            "  Stage '{}': {}",
            result.stage,
            &result.output[..result.output.len().min(50)]
        );
    }
    println!();

    // -------------------------------------------------------------------------
    // 9. Security Demo Functions
    // -------------------------------------------------------------------------
    println!("9. Security Demo: Compile-Time Enforcement");
    println!("------------------------------------------\n");

    println!("Function signatures encode security requirements:\n");

    println!("  // Only works with full access");
    println!("  fn full_access_agent<S: HasDatabase + HasLLM>(scope: &S) -> ...");
    let full = FullScope::new(MockDatabase::new(), MockLLM::new("model"));
    let result = security_demo::full_access_agent(&full, "task").unwrap();
    println!("  Result: {}\n", &result[..result.len().min(50)]);

    println!("  // Only needs read-only access");
    println!("  fn read_only_agent<S: HasReadOnlyDatabase>(scope: &S) -> ...");
    let ro = ReadOnlyScope::new(MockDatabase::new());
    let result = security_demo::read_only_agent(&ro, "users").unwrap();
    println!("  Result: {:?}\n", result);

    // -------------------------------------------------------------------------
    // 10. Comparison Summary
    // -------------------------------------------------------------------------
    println!("10. Comparison: Runtime vs Compile-Time Safety");
    println!("----------------------------------------------\n");

    println!("  | Aspect            | Python/JS (Runtime)     | Rust (Compile-Time)      |");
    println!("  |-------------------|-------------------------|--------------------------|");
    println!("  | Isolation         | WASM sandbox, MicroVM   | Lifetime annotations     |");
    println!("  | Thread safety     | Locks + hope            | Send/Sync proofs         |");
    println!("  | Capability control| Runtime checks          | Trait bounds             |");
    println!("  | Overhead          | Always present          | Zero at runtime          |");
    println!(
        "  | Guarantees        | \"Should work\"           | \"Will work or won't compile\" |"
    );
    println!();

    // -------------------------------------------------------------------------
    // Exercises
    // -------------------------------------------------------------------------
    println!("=== Exercises ===\n");

    println!("Exercise 1: Lifetime Error (Uncomment to see compile error)");
    println!("------------------------------------------------------------");
    println!("  // let agent;");
    println!("  // {{");
    println!("  //     let scope = FullScope::new(...);");
    println!("  //     agent = ScopedAgent::new(&scope, \"test\");");
    println!("  // }}");
    println!("  // agent.execute(\"task\");  // ERROR: scope doesn't live long enough");
    println!();

    println!("Exercise 2: Wrong Capabilities (Uncomment to see compile error)");
    println!("----------------------------------------------------------------");
    println!("  // let scope = ReadOnlyScope::new(MockDatabase::new());");
    println!("  // let agent = ScopedAgent::new(&scope, \"test\");");
    println!("  // agent.execute(\"task\");  // ERROR: HasLLM not satisfied");
    println!();

    println!("Exercise 3: Create Custom Scope");
    println!("-------------------------------");
    println!("  Implement a scope with only LLM access (no database)");
    println!("  Hint: Create LLMOnlyScope implementing only HasLLM");
    println!();

    println!("Exercise 4: Async Agents");
    println!("------------------------");
    println!("  Extend ScopedAgent with async methods using tokio");
    println!("  Ensure proper Send bounds for spawning tasks");
    println!();

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    println!("=== Session 18.5 Complete ===\n");

    println!("Key Takeaways:");
    println!("  1. Lifetimes are sandboxes — agents cannot access outside their scope");
    println!("  2. Send/Sync are proofs — compiler verifies thread safety");
    println!("  3. Traits are capabilities — bound what agents can access");
    println!("  4. Zero cost — all checks at compile time, no runtime overhead");
    println!();

    println!("The borrow checker IS the sandbox!");
    println!("  • No WASM needed");
    println!("  • No MicroVMs needed");
    println!("  • No runtime checks needed");
    println!("  • Compiler proves safety BEFORE your code even runs");
    println!();

    println!("Next: Session 19 — Agent composition patterns");
}
