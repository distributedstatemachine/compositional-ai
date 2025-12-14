//! Session 5: Parallel Agents - Runtime Tensor Product
//!
//! Run with: cargo run --example session5_parallel
//!
//! This example demonstrates:
//! - The Agent trait for async computations
//! - ParallelAgents for concurrent execution
//! - Fan-out (broadcast input to multiple agents)
//! - Combiners for merging results (fan-in)
//! - The tensor product at runtime

use compositional_core::parallel::{
    Agent, Collect, Combiner, Concat, ConcatWith, First, ParallelAgents, Vote,
};
use compositional_core::CoreError;
use std::future::Future;
use std::time::Duration;

// ============================================================================
// Example Agents
// ============================================================================

/// An agent that doubles its input
#[derive(Clone)]
struct DoubleAgent;

impl Agent for DoubleAgent {
    type Input = i32;
    type Output = i32;
    type Error = CoreError;

    fn run(
        &self,
        input: Self::Input,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
        async move { Ok(input * 2) }
    }
}

/// An agent that adds a prefix to strings
#[derive(Clone)]
struct PrefixAgent {
    prefix: String,
}

impl Agent for PrefixAgent {
    type Input = String;
    type Output = String;
    type Error = CoreError;

    fn run(
        &self,
        input: Self::Input,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
        let prefix = self.prefix.clone();
        async move { Ok(format!("{}: {}", prefix, input)) }
    }
}

/// An agent that simulates a slow operation
#[derive(Clone)]
struct SlowAgent {
    name: String,
    delay_ms: u64,
}

impl Agent for SlowAgent {
    type Input = String;
    type Output = String;
    type Error = CoreError;

    fn run(
        &self,
        input: Self::Input,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
        let name = self.name.clone();
        let delay = self.delay_ms;
        async move {
            tokio::time::sleep(Duration::from_millis(delay)).await;
            Ok(format!("[{}] processed: {}", name, input))
        }
    }
}

/// An agent that classifies sentiment (returns fixed value for demo)
#[derive(Clone)]
struct SentimentAgent {
    bias: String, // "positive", "negative", or "neutral"
}

impl Agent for SentimentAgent {
    type Input = String;
    type Output = String;
    type Error = CoreError;

    fn run(
        &self,
        _input: Self::Input,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
        let bias = self.bias.clone();
        async move { Ok(bias) }
    }
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), CoreError> {
    println!("=== Session 5: Parallel Agents ===\n");

    // -------------------------------------------------------------------------
    // Basic Parallel Execution
    // -------------------------------------------------------------------------
    println!("1. Basic Parallel Execution");
    println!("---------------------------");

    let agents = ParallelAgents::new()
        .with(DoubleAgent)
        .with(DoubleAgent)
        .with(DoubleAgent);

    let inputs = vec![1, 2, 3];
    println!("Inputs: {:?}", inputs);

    let results = agents.run_all(inputs).await?;
    println!("Results (doubled): {:?}", results);
    println!();

    // -------------------------------------------------------------------------
    // Fan-Out (Broadcast)
    // -------------------------------------------------------------------------
    println!("2. Fan-Out (Broadcast)");
    println!("----------------------");

    let researchers = ParallelAgents::new()
        .with(PrefixAgent {
            prefix: "Technical".into(),
        })
        .with(PrefixAgent {
            prefix: "Academic".into(),
        })
        .with(PrefixAgent {
            prefix: "Popular".into(),
        });

    let query = "What is category theory?".to_string();
    println!("Query: \"{}\"", query);
    println!("Broadcasting to 3 agents...");

    let findings = researchers.fan_out(query).await?;
    for finding in &findings {
        println!("  {}", finding);
    }
    println!();

    // -------------------------------------------------------------------------
    // Tensor Product of Agent Groups
    // -------------------------------------------------------------------------
    println!("3. Tensor Product of Agent Groups");
    println!("---------------------------------");

    let group1 = ParallelAgents::new().with(DoubleAgent).with(DoubleAgent);

    let group2 = ParallelAgents::new().with(DoubleAgent);

    println!("Group 1: 2 agents");
    println!("Group 2: 1 agent");

    let combined = group1.tensor(group2);
    println!("Combined (tensor): {} agents", combined.len());

    let results = combined.run_all(vec![10, 20, 30]).await?;
    println!("Results: {:?}", results);
    println!();

    // -------------------------------------------------------------------------
    // Combiners: Concat
    // -------------------------------------------------------------------------
    println!("4. Combiners: Concat");
    println!("--------------------");

    let concat = Concat;
    let strings = vec![
        "First result".to_string(),
        "Second result".to_string(),
        "Third result".to_string(),
    ];

    println!("Inputs: {:?}", strings);
    println!("Combined:\n{}", concat.combine(strings));
    println!();

    // -------------------------------------------------------------------------
    // Combiners: ConcatWith
    // -------------------------------------------------------------------------
    println!("5. Combiners: ConcatWith");
    println!("------------------------");

    let concat_csv = ConcatWith::new(", ");
    let items = vec!["apple".into(), "banana".into(), "cherry".into()];

    println!("Inputs: {:?}", items);
    println!("Combined: {}", concat_csv.combine(items));
    println!();

    // -------------------------------------------------------------------------
    // Combiners: Vote (Majority)
    // -------------------------------------------------------------------------
    println!("6. Combiners: Vote");
    println!("------------------");

    // Simulate multiple sentiment classifiers
    let classifiers = ParallelAgents::new()
        .with(SentimentAgent {
            bias: "positive".into(),
        })
        .with(SentimentAgent {
            bias: "positive".into(),
        })
        .with(SentimentAgent {
            bias: "negative".into(),
        })
        .with(SentimentAgent {
            bias: "positive".into(),
        })
        .with(SentimentAgent {
            bias: "neutral".into(),
        });

    let votes = classifiers.fan_out("I love this product!".into()).await?;
    println!("Votes: {:?}", votes);

    let vote = Vote;
    let winner = vote.combine(votes);
    println!("Winner (majority): {:?}", winner);
    println!();

    // -------------------------------------------------------------------------
    // Combiners: First and Collect
    // -------------------------------------------------------------------------
    println!("7. Other Combiners");
    println!("------------------");

    let numbers = vec![10, 20, 30, 40];

    let first = First;
    println!(
        "First of {:?}: {:?}",
        numbers.clone(),
        first.combine(numbers.clone())
    );

    let collect = Collect;
    println!(
        "Collect {:?}: {:?}",
        numbers.clone(),
        collect.combine(numbers)
    );
    println!();

    // -------------------------------------------------------------------------
    // Parallel Speedup Demo
    // -------------------------------------------------------------------------
    println!("8. Parallel Speedup");
    println!("-------------------");

    let slow_agents = ParallelAgents::new()
        .with(SlowAgent {
            name: "A".into(),
            delay_ms: 100,
        })
        .with(SlowAgent {
            name: "B".into(),
            delay_ms: 100,
        })
        .with(SlowAgent {
            name: "C".into(),
            delay_ms: 100,
        });

    println!("Running 3 agents (each 100ms) in parallel...");
    let start = std::time::Instant::now();
    let results = slow_agents.fan_out("data".into()).await?;
    let elapsed = start.elapsed();

    for r in &results {
        println!("  {}", r);
    }
    println!("Total time: {:?} (should be ~100ms, not 300ms)", elapsed);

    println!("\n=== Session 5 Complete ===");
    Ok(())
}
