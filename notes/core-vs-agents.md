# Core vs Agents: Separation of Concerns

## Guiding Principle

**Core**: General-purpose compositional abstractions that could be used for ANY domain (agents, autodiff, probability, NLP, games).

**Agents**: AI agent-specific implementations that build ON TOP of core abstractions.

## The Test

Ask: "Would this be useful for probability, autodiff, or games too?"
- **Yes** → goes in `core`
- **No, only for AI agents** → goes in `agents`

---

## What's Already in Core (Keep)

| Module | Purpose | Used By |
|--------|---------|---------|
| `shape.rs` | Type-level tensor descriptions | All crates |
| `diagram.rs` | String diagram data model | diff, agents |
| `cat.rs` | Categories, functors, coproducts | All crates |
| `capability.rs` | Yoneda-style Request/Handles | agents |
| `parallel.rs` | Agent trait, ParallelAgents | agents, games |
| `tracing.rs` | Zero-cost Computation tracing | All crates |
| `operad.rs` | Arity enforcement | agents |
| `error.rs` | Composition failures | All crates |

**Verdict**: All current core modules are general-purpose. Keep them.

---

## What Goes in Core

### Already There (Good)

```
core/
├── capability.rs    # Request, Handles<R>, CapabilityScope, CapabilityError
│                    # HasDatabase, HasLLM, HasFileSystem, HasReadOnlyDatabase
├── parallel.rs      # Agent trait, ParallelAgents, Combiner
├── tracing.rs       # Computation trait, Traced<C, ENABLED>, TraceNode
├── operad.rs        # Operation, WiringPlan, OperadError
├── diagram.rs       # Diagram<O>, Node, Port, Edge
├── shape.rs         # Shape, TypeId
├── cat.rs           # FiniteCategory, Morphism, Scope, Coproduct
└── error.rs         # CoreError
```

### Should Add to Core

Nothing new needed. The current abstractions are sufficient:

- `Request` + `Handles<R>` — generic request/response pattern
- `CapabilityScope` — generic capability registry
- `Computation` — generic async computation abstraction
- `Agent` — generic parallel execution unit
- `Operation` + `WiringPlan` — generic arity-enforced composition
- `Diagram` — generic graph structure

---

## What Goes in Agents

### Agent-Specific Request Types

```rust
// agents/src/requests.rs

/// LLM completion request (agent-specific)
pub struct LlmComplete {
    pub messages: Vec<Message>,
    pub tools: Option<Vec<ToolSchema>>,
    pub max_tokens: usize,
    pub temperature: f32,
}

impl Request for LlmComplete {
    type Response = LlmResponse;
    fn name() -> &'static str { "LlmComplete" }
}

/// Tool invocation request (agent-specific)
pub struct ToolInvoke {
    pub tool_name: String,
    pub args: serde_json::Value,
}

impl Request for ToolInvoke {
    type Response = ToolResult;
    fn name() -> &'static str { "ToolInvoke" }
}
```

**Why in agents**: These are specific to AI agent use cases. Probability crate doesn't need `LlmComplete`.

### LLM Client Implementations

```rust
// agents/src/llm/mod.rs

pub mod anthropic;
pub mod openai;

/// Anthropic Claude client
pub struct AnthropicClient {
    api_key: String,
    model: String,
}

impl Capability for AnthropicClient { ... }
impl Handles<LlmComplete> for AnthropicClient { ... }

/// OpenAI client
pub struct OpenAIClient { ... }
impl Handles<LlmComplete> for OpenAIClient { ... }
```

**Why in agents**: Specific to AI agents. Other crates don't need LLM clients.

### Tool Abstraction

```rust
// agents/src/tool.rs

/// A tool wraps a capability for LLM function calling
pub struct Tool {
    pub name: String,
    pub description: String,
    pub schema: ToolSchema,
    handler: Box<dyn AnyToolHandler>,
}

/// Registry of tools using CapabilityScope
pub struct ToolRegistry {
    scope: CapabilityScope,  // From core!
    tools: HashMap<String, Tool>,
}

impl ToolRegistry {
    /// Register a tool backed by a Handles<R> capability
    pub fn register<C, R>(&mut self, name: &str, desc: &str, cap: Arc<C>)
    where
        C: Handles<R> + 'static,
        R: Request + JsonSchema;

    /// Get tool schemas for LLM function calling
    pub fn schemas(&self) -> Vec<ToolSchema>;

    /// Invoke tool (dispatches to CapabilityScope)
    pub fn invoke(&self, name: &str, args: Value) -> Result<Value, CapabilityError>;
}
```

**Why in agents**: Tools are an agent-specific concept (LLM function calling). Core's `Handles<R>` is the generic pattern; `Tool` is the agent-specific wrapper.

### Agent Loop

```rust
// agents/src/agent.rs

use compositional_core::{Computation, CoreError};

/// The core agent loop
pub struct AgentLoop<S> {
    scope: Arc<S>,
    tools: ToolRegistry,
    config: AgentConfig,
}

impl<S> Computation for AgentLoop<S>
where
    S: Handles<LlmComplete> + Send + Sync + 'static,
{
    type Input = AgentTask;
    type Output = AgentResult;

    async fn run(&self, input: Self::Input) -> Result<Self::Output, CoreError> {
        // Agent-specific logic:
        // 1. Build messages
        // 2. Call LLM with tools
        // 3. If tool call, execute and loop
        // 4. Return final answer
    }
}
```

**Why in agents**: The agent loop is agent-specific. Core provides `Computation` trait; agents provides `AgentLoop` implementation.

### Multi-Agent Orchestration

```rust
// agents/src/orchestrator.rs

use compositional_core::{WiringPlan, OperadError, ParallelAgents};

/// Multi-agent orchestrator using operadic composition
pub struct Orchestrator {
    plan: WiringPlan,  // From core!
    agents: Vec<Arc<dyn AgentRunner>>,
}

impl Orchestrator {
    /// Validate wiring before execution
    pub fn validate(&self) -> Result<(), OperadError> {
        self.plan.validate()  // Uses core's operad validation
    }

    /// Execute with parallel tool calls
    pub async fn execute(&self, input: &str) -> Result<String, AgentError> {
        // Use core's ParallelAgents for concurrent execution
    }
}
```

**Why in agents**: Orchestration is agent-specific, but it USES core's `WiringPlan` and `ParallelAgents`.

### Execution Trace

```rust
// agents/src/trace.rs

use compositional_core::{Diagram, Node, TraceNode};

/// Agent-specific operation for diagrams
#[derive(Clone, Debug)]
pub enum AgentOp {
    LlmCall { model: String, tokens: usize },
    ToolCall { name: String, duration_ms: u64 },
    Decision { branch: String },
}

/// Convert agent execution to diagram
pub fn trace_to_diagram(trace: &AgentTrace) -> Diagram<AgentOp> {
    // Build diagram from trace
}
```

**Why in agents**: `AgentOp` is agent-specific. Core provides generic `Diagram<O>`; agents provides `AgentOp` as the operation type.

---

## Summary: The Boundary

```
┌─────────────────────────────────────────────────────────────────┐
│                           CORE                                  │
│  General-purpose compositional abstractions                     │
│                                                                 │
│  Request, Handles<R>     - Generic request/response pattern     │
│  CapabilityScope         - Generic capability registry          │
│  Computation             - Generic async abstraction            │
│  Agent, ParallelAgents   - Generic parallel execution           │
│  Operation, WiringPlan   - Generic arity enforcement            │
│  Diagram<O>              - Generic graph structure              │
│  Traced<C, ENABLED>      - Generic zero-cost tracing            │
│  Shape                   - Generic type descriptions            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ uses
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          AGENTS                                 │
│  AI agent-specific implementations                              │
│                                                                 │
│  LlmComplete, ToolInvoke - Agent-specific Request types         │
│  AnthropicClient         - Handles<LlmComplete> for Claude      │
│  Tool, ToolRegistry      - LLM function calling wrapper         │
│  AgentLoop               - Computation impl for agent loop      │
│  Orchestrator            - Multi-agent using WiringPlan         │
│  AgentOp                 - Agent-specific Diagram operation     │
│  AgentTrace              - Agent-specific TraceNode data        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Decision Table

| Component | In Core? | In Agents? | Reason |
|-----------|----------|------------|--------|
| `Request` trait | ✓ | | Generic pattern |
| `Handles<R>` trait | ✓ | | Generic pattern |
| `CapabilityScope` | ✓ | | Generic registry |
| `LlmComplete` request | | ✓ | Agent-specific |
| `ToolInvoke` request | | ✓ | Agent-specific |
| `AnthropicClient` | | ✓ | Agent-specific |
| `Tool` wrapper | | ✓ | Agent-specific |
| `ToolRegistry` | | ✓ | Uses core's CapabilityScope |
| `Computation` trait | ✓ | | Generic async pattern |
| `AgentLoop` | | ✓ | Implements Computation |
| `Agent` trait | ✓ | | Generic parallel unit |
| `ParallelAgents` | ✓ | | Generic parallel execution |
| `Operation` | ✓ | | Generic arity |
| `WiringPlan` | ✓ | | Generic composition |
| `Orchestrator` | | ✓ | Uses WiringPlan |
| `Diagram<O>` | ✓ | | Generic graph |
| `AgentOp` | | ✓ | Agent-specific operation |
| `Traced<C, E>` | ✓ | | Generic tracing |
| `AgentTrace` | | ✓ | Agent-specific trace data |
| `HasDatabase` | ✓ | | Standard capability trait |
| `HasLLM` | ✓ | | Standard capability trait |

---

## File Structure

```
crates/
├── core/
│   └── src/
│       ├── lib.rs
│       ├── capability.rs   # Request, Handles, CapabilityScope, Has* traits
│       ├── parallel.rs     # Agent, ParallelAgents, Combiner
│       ├── tracing.rs      # Computation, Traced, TraceNode
│       ├── operad.rs       # Operation, WiringPlan
│       ├── diagram.rs      # Diagram, Node, Port
│       ├── shape.rs        # Shape, TypeId
│       ├── cat.rs          # FiniteCategory, Scope, Coproduct
│       └── error.rs        # CoreError
│
└── agents/
    └── src/
        ├── lib.rs
        ├── requests.rs     # LlmComplete, ToolInvoke (impl Request)
        ├── llm/
        │   ├── mod.rs
        │   ├── anthropic.rs  # AnthropicClient (impl Handles<LlmComplete>)
        │   └── openai.rs     # OpenAIClient (impl Handles<LlmComplete>)
        ├── tool.rs         # Tool, ToolRegistry (wraps CapabilityScope)
        ├── agent.rs        # AgentLoop (impl Computation), AgentConfig
        ├── orchestrator.rs # Orchestrator (uses WiringPlan)
        └── trace.rs        # AgentOp, AgentTrace, trace_to_diagram
```

---

## Key Insight

**Core provides the PATTERNS, Agents provides the IMPLEMENTATIONS.**

- Core: "Here's how to do request/response" (`Request` + `Handles<R>`)
- Agents: "Here's a request for LLM completion" (`LlmComplete`)

- Core: "Here's how to compose with arity" (`WiringPlan`)
- Agents: "Here's how to orchestrate multiple agents" (`Orchestrator`)

- Core: "Here's how to trace computation" (`Traced<C, ENABLED>`)
- Agents: "Here's what agent traces look like" (`AgentTrace`)

This separation keeps core reusable across ALL domain crates while allowing agents to build agent-specific functionality on top.
