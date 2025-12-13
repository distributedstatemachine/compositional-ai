# Improvements Over Agentica

*Concrete enhancements enabled by categorical foundations + Rust's type system.*

---

## Summary Table

| Agentica Feature | Current (Python) | Rust Improvement | Course Session |
|------------------|------------------|------------------|----------------|
| Type safety | Runtime `Literal[]` checks | Compile-time enums | 1-3 |
| Scope | `Dict[str, Any]` | Trait bounds | 3.5 |
| Scope merging | Dict update (silent conflicts) | Trait composition | 3.5 |
| Tool discovery | Runtime introspection | Compile-time traits | 3.5 (Yoneda) |
| Composition | Implicit chaining | Type-level `Then<A,B>` | 4-5 |
| Parallelism | `asyncio.gather` | `Send + Sync` proofs | 5 |
| Sandboxing | WASM + microVM | Borrow checker | 3 (Functors) |
| Arity checks | Runtime validation | GATs + const generics | 19 |
| Credit assignment | Not supported | Backprop over traces | 9 |
| Traces | Always-on overhead | Zero-cost const toggle | 6 |

---

## Detailed Improvements

### 1. Compile-Time Type Safety

**Agentica:**
```python
@agentic()
async def classify(text: str) -> Literal["positive", "negative", "neutral"]:
    """Runtime validated — crashes in production if LLM returns "unknown"."""
    ...
```

**Rust:**
```rust
enum Sentiment { Positive, Negative, Neutral }

async fn classify(text: &str) -> Result<Sentiment, AgentError> {
    // Compile-time enforced — can't return invalid variant
    // LLM output parsing is explicit, errors handled
}
```

**Benefit:** Invalid states are unrepresentable. No runtime type validation needed.

---

### 2. Trait-Based Capabilities Instead of Dict Scope

**Agentica:**
```python
@agentic(scope={"db": database, "cache": cache})
async def agent(api: APIClient):
    # Runtime discovery of what's in scope
    # No compile-time guarantee db exists
    ...
```

**Rust:**
```rust
async fn agent<S>(scope: &S, api: &APIClient) -> Result<Output, Error>
where
    S: HasDatabase + HasCache,  // Compile-time requirement
{
    // Compiler guarantees db() and cache() exist
    scope.db().query(...).await?;
    scope.cache().get(...).await?;
}
```

**Benefit:** Missing capabilities are compile errors, not runtime crashes.

---

### 3. Safe Scope Composition

**Agentica:**
```python
# Silent conflict — second wins
merged = {**global_scope, **local_scope}
```

**Rust:**
```rust
struct Combined<A, B> { a: A, b: B }

// Explicit delegation — no hidden conflicts
impl<A: HasDatabase, B> HasDatabase for Combined<A, B> {
    fn db(&self) -> &dyn Database { self.a.db() }
}

impl<A, B: HasCache> HasCache for Combined<A, B> {
    fn cache(&self) -> &dyn Cache { self.b.cache() }
}

// Conflict detection at compile time:
// If both A and B implement HasDatabase, you must choose which to expose
```

**Benefit:** Scope composition is explicit. Conflicts are visible.

---

### 4. Zero-Cost Isolation (No Sandbox)

**Agentica:**
```python
# Requires WASM sandbox + microVM for security
# Significant runtime overhead
# Objects proxied via RPC
```

**Rust:**
```rust
struct AgentView<'a> {
    // Agent only sees what we give it
    db: &'a dyn ReadOnlyDatabase,  // Can't write
    // private_data not included — invisible to agent
}

impl<'a> AgentView<'a> {
    fn new(full_scope: &'a FullScope) -> Self {
        Self { db: &full_scope.db }
    }
}

// Agent literally cannot access what's not in AgentView
// Enforced by compiler, not sandbox
```

**Benefit:** Security via types, not runtime isolation. Zero overhead.

---

### 5. Proven Parallel Safety

**Agentica:**
```python
# Hope these don't race
results = await asyncio.gather(
    agent1(scope),
    agent2(scope),
)
```

**Rust:**
```rust
async fn parallel<S>(scope: Arc<S>) -> (A, B)
where
    S: Send + Sync,  // Compiler PROVES this is safe
{
    tokio::join!(
        agent1(scope.clone()),
        agent2(scope.clone()),
    )
}

// If S contains interior mutability unsafely, this WON'T COMPILE
```

**Benefit:** Data races are compile errors.

---

### 6. Type-Level Composition

**Agentica:**
```python
# Composition is implicit control flow
result1 = await agent1(input)
result2 = await agent2(result1)  # Hope types match
```

**Rust:**
```rust
// Composition is explicit and type-checked
type Pipeline = Then<Agent1, Agent2>;
// Only compiles if Agent1::Output == Agent2::Input

let pipeline: Pipeline = Then { first: agent1, second: agent2 };
let result = pipeline.run(input).await?;
```

**Benefit:** Pipeline structure is visible, verified, optimizable.

---

### 7. Compile-Time Arity Enforcement

**Agentica:**
```python
# Runtime error if wrong number of sub-agents
async def orchestrator(agents: list[Agent]):
    if len(agents) != 3:
        raise ValueError("Expected 3 agents")
```

**Rust:**
```rust
// Compile-time arity via tuple types
async fn orchestrator(
    research: ResearchAgent,
    analysis: AnalysisAgent,
    summary: SummaryAgent,  // Exactly 3, enforced by signature
) -> FinalReport {
    ...
}

// Or via const generics:
struct Orchestrator<const N: usize>;

impl Orchestrator<3> {
    async fn run(&self, agents: [Box<dyn Agent>; 3]) -> Report { ... }
}

// Orchestrator<3>::run with 2 agents WON'T COMPILE
```

**Benefit:** Arity mismatches caught before running.

---

### 8. Zero-Cost Tracing

**Agentica:**
```python
# Tracing always enabled, always costs tokens/time
# No way to disable in production
```

**Rust:**
```rust
// Const generic toggle — zero cost when disabled
struct Traced<C, const ENABLED: bool>(C);

impl<C: Computation> Computation for Traced<C, false> {
    // No tracing code generated
    async fn run(&self, input: Input) -> Output {
        self.0.run(input).await
    }
}

impl<C: Computation> Computation for Traced<C, true> {
    // Full tracing
    async fn run(&self, input: Input) -> (Output, Trace) {
        let start = Instant::now();
        let result = self.0.run(input).await;
        (result, Trace { duration: start.elapsed() })
    }
}

// Compile-time selection:
#[cfg(debug_assertions)]
type MyAgent = Traced<AgentImpl, true>;

#[cfg(not(debug_assertions))]
type MyAgent = Traced<AgentImpl, false>;  // Zero overhead in release
```

**Benefit:** Debug builds trace everything; release builds have zero overhead.

---

### 9. Backprop-Style Credit Assignment

**Agentica:** Not supported — when an agent fails, no way to know which tool caused it.

**Rust:**
```rust
// Trace is a Diagram<AgentOp>
// Apply backward pass to attribute credit

fn credit_assignment(
    trace: &Diagram<AgentOp>,
    outcome: Outcome,
) -> HashMap<NodeId, f32> {
    let mut credits = HashMap::new();

    // Reverse topological order (like backprop)
    for node in trace.reverse_topo() {
        let credit = match &node.op {
            AgentOp::LLMCall { .. } => compute_llm_credit(&node, &outcome),
            AgentOp::ToolCall { name, .. } => compute_tool_credit(name, &outcome),
            AgentOp::Branch { .. } => propagate_through_branch(&node),
        };
        credits.insert(node.id, credit);
    }
    credits
}
```

**Benefit:** When something fails (or succeeds), you know exactly which operations contributed.

---

### 10. Lifetime-Scoped Resources

**Agentica:**
```python
# GC manages lifetimes — objects can leak across boundaries
# Agent might hold onto resources after scope "closes"
```

**Rust:**
```rust
struct Agent<'scope, S> {
    scope: &'scope S,  // Borrowed, not owned
}

impl<'scope, S> Agent<'scope, S> {
    async fn call(&self) -> Output {
        // self.scope valid for entire call
        // Cannot store references that outlive 'scope
    }
}

fn run() {
    let resources = Resources::new();
    {
        let agent = Agent { scope: &resources };
        agent.call().await;
    }  // agent dropped, scope borrow ends

    // resources still valid here, agent gone
    // No leaks, no dangling refs — compiler enforced
}
```

**Benefit:** Resource lifetimes are explicit and enforced.

---

## What We're NOT Doing

Some Agentica features don't need Rust equivalents:

1. **Dynamic tool registration** — Rust's traits make this compile-time
2. **Schema inference** — Types ARE the schema
3. **Forbidden function checks** — Borrow checker handles this
4. **Runtime type coercion** — No implicit conversions in Rust

---

## Implementation Priority

### High Value, Low Effort
1. Trait-based capabilities (replaces scope dict)
2. Type-level composition (`Then<A, B>`)
3. Compile-time arity via tuples

### High Value, Medium Effort
4. Zero-cost tracing toggle
5. Parallel safety via `Send + Sync`
6. Lifetime-scoped agents

### High Value, High Effort
7. Const generic shapes
8. GAT-based operads
9. Backprop credit assignment

---

## Conclusion

Agentica's core insight is correct: **composition is the right abstraction for agents**.

But Python forces runtime checks for guarantees Rust provides at compile time. The Rust version isn't a port — it's a redesign where:

- **Types encode capabilities**
- **Composition is type-checked**
- **Safety is proven, not hoped**

The course builds the categorical foundations. This document shows the practical payoff.
