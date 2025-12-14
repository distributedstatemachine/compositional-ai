//! # Compositional Agents
//!
//! A Rust-native AI agent framework with compile-time safety guarantees.
//!
//! This crate builds on `compositional-core` to provide:
//!
//! - **Type-safe tools**: Tools implement `Handles<R>` from core's capability system
//! - **Traceable execution**: Agent loops implement `Computation` for zero-cost tracing
//! - **Arity enforcement**: Multi-agent pipelines validated via operads
//! - **Parallel safety**: `Send + Sync` proofs for concurrent agent execution
//!
//! ## Quick Start
//!
//! ```rust
//! use compositional_agents::{
//!     SimpleAgent, ToolRegistry,
//!     tool::{calculator_tool, mock_search_tool},
//!     llm::MockLlmClient,
//! };
//! use std::sync::Arc;
//!
//! // Create an LLM client
//! let llm = Arc::new(MockLlmClient::new("test-model"));
//!
//! // Register tools
//! let mut tools = ToolRegistry::new();
//! tools.register(calculator_tool());
//! tools.register(mock_search_tool());
//!
//! // Create agent and run
//! let agent = SimpleAgent::with_tools(llm, tools);
//! let result = agent.run("What is 2 + 2?").unwrap();
//! println!("Response: {}", result.response);
//! ```
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    compositional-agents                         │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                 │
//! │  requests.rs    LlmRequest, ToolInvoke (impl Request)           │
//! │  llm.rs         MockLlmClient (impl Handles<LlmRequest>)        │
//! │  tool.rs        Tool, ToolRegistry (wraps CapabilityScope)      │
//! │  agent.rs       AgentLoop (impl Computation)                    │
//! │  orchestrator.rs Orchestrator (uses WiringPlan)                 │
//! │  trace.rs       AgentOp, AgentTrace (Diagram<AgentOp>)          │
//! │                                                                 │
//! └───────────────────────────┬─────────────────────────────────────┘
//!                             │ uses
//!                             ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    compositional-core                           │
//! │  Request, Handles<R>, CapabilityScope, Computation, WiringPlan  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Compile-Time Guarantees
//!
//! Unlike Python frameworks, this crate provides:
//!
//! - **Capability checking at compile time**: Missing capabilities are compile errors
//! - **Thread safety proofs**: `Send + Sync` bounds enforced by rustc
//! - **Arity validation**: Wrong number of tool inputs caught before execution
//! - **Zero-cost tracing**: Const generics compile out tracing in release builds

pub mod agent;
pub mod hooks;
pub mod llm;
pub mod orchestrator;
pub mod requests;
pub mod tool;
pub mod trace;

// Re-export key types at crate root
pub use agent::{AgentConfig, AgentError, AgentLoop, AgentResult, AgentTask, SimpleAgent};
pub use hooks::{AgentHook, CompositeHook, LoggingHook, NullHook};
pub use llm::{DeterministicMockLlm, LlmClient, MockLlmClient};
pub use orchestrator::{
    fanout_pipeline, sequential_pipeline, AgentRunner, AgentWrapper, Orchestrator,
    OrchestratorResult, PipelineStage,
};
pub use requests::{
    Document, LlmRequest, LlmResponse, Message, Role, ToolCallRequest, ToolChoice, ToolInvoke,
    ToolResult, ToolSchema,
};
pub use tool::{ToolDef, ToolRegistry};
pub use trace::{AgentOp, AgentTrace, TraceEvent, TraceSummary};

// Re-export core types commonly used with agents
pub use compositional_core::{
    Capability, CapabilityError, CapabilityScope, Handles, HasDatabase, HasLLM, Request,
};
