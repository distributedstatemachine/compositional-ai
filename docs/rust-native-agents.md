# Rust-Native Agent Framework

*How categorical foundations + Rust's type system enable a fundamentally different agent architecture.*

---

## Why Rust Changes Everything

Agentica (Python) fights the language to get safety:
- Runtime type validation
- WASM sandboxing for isolation
- Dict-based scope with introspection
- Hope-based parallelism

Rust gives these properties **for free** via the type system:
- Compile-time type enforcement
- Ownership/borrowing for isolation
- Trait bounds for capabilities
- `Send + Sync` for safe concurrency

This document outlines a Rust-native agent framework built on the compositional foundations from the course.

---

## Core Architecture

### 1. Type-Level Computation Graphs

Instead of runtime `Diagram<AgentOp>`, encode composition in types:

```rust
/// A computation with typed input/output
trait Computation {
    type Input;
    type Output;

    async fn run(&self, input: Self::Input) -> Result<Self::Output, AgentError>;
}

/// Sequential composition: A then B
/// Only valid if A::Output == B::Input (enforced at compile time)
struct Then<A, B> {
    first: A,
    second: B,
}

impl<A, B> Computation for Then<A, B>
where
    A: Computation,
    B: Computation<Input = A::Output>,
{
    type Input = A::Input;
    type Output = B::Output;

    async fn run(&self, input: Self::Input) -> Result<Self::Output, AgentError> {
        let mid = self.first.run(input).await?;
        self.second.run(mid).await
    }
}

/// Parallel composition: A ⊗ B (monoidal product)
struct Tensor<A, B> {
    left: A,
    right: B,
}

impl<A, B> Computation for Tensor<A, B>
where
    A: Computation + Send,
    B: Computation + Send,
    A::Input: Send,
    B::Input: Send,
{
    type Input = (A::Input, B::Input);
    type Output = (A::Output, B::Output);

    async fn run(&self, input: Self::Input) -> Result<Self::Output, AgentError> {
        let (a, b) = tokio::join!(
            self.left.run(input.0),
            self.right.run(input.1),
        );
        Ok((a?, b?))
    }
}
```

**Key insight:** Mismatched composition is a *compile error*, not a runtime crash.

```rust
// This compiles:
type ValidPipeline = Then<TextEncoder, VectorSearch>;
// TextEncoder::Output = Vector, VectorSearch::Input = Vector ✓

// This WON'T compile:
type InvalidPipeline = Then<TextEncoder, ImageClassifier>;
// TextEncoder::Output = Vector, ImageClassifier::Input = Image ✗
// Error: type mismatch
```

---

### 2. Const Generic Shapes

Shapes verified at compile time via const generics:

```rust
/// A tensor with compile-time shape
#[derive(Clone)]
struct Tensor<T, const N: usize> {
    data: [T; N],
}

/// Type aliases for common shapes
type Scalar<T> = Tensor<T, 1>;
type Vector<T, const N: usize> = Tensor<T, N>;

/// A linear layer with compile-time dimension checking
struct Linear<const IN: usize, const OUT: usize> {
    weights: [[f32; IN]; OUT],
    bias: [f32; OUT],
}

impl<const IN: usize, const OUT: usize> Linear<IN, OUT> {
    fn forward(&self, input: Vector<f32, IN>) -> Vector<f32, OUT> {
        // Implementation...
        // Shape mismatch is IMPOSSIBLE — won't compile
    }
}

// Composing layers — dimensions must chain:
type MLP = Then<Linear<784, 512>, Then<Linear<512, 256>, Linear<256, 10>>>;
// Compile-time verified: 784 → 512 → 256 → 10
```

---

### 3. Trait-Based Capabilities (Not Dict Scope)

Instead of `scope = {"db": db, "cache": cache}`, use trait bounds:

```rust
/// Capability traits — what a scope can provide
trait HasDatabase {
    type DB: Database;
    fn db(&self) -> &Self::DB;
}

trait HasCache {
    type Cache: Cache;
    fn cache(&self) -> &Self::Cache;
}

trait HasLLM {
    type LLM: LanguageModel;
    fn llm(&self) -> &Self::LLM;
}

/// Agent declares required capabilities as trait bounds
async fn research_agent<S>(scope: &S, query: &str) -> Result<Report, AgentError>
where
    S: HasDatabase + HasCache + HasLLM,
{
    // Compiler guarantees these methods exist
    let results = scope.db().search(query).await?;
    let cached = scope.cache().get(query).await;
    let summary = scope.llm().summarize(&results).await?;
    Ok(Report { results, summary })
}

/// Concrete scope implementing capabilities
struct ProductionScope {
    db: PostgresDB,
    cache: RedisCache,
    llm: ClaudeClient,
}

impl HasDatabase for ProductionScope {
    type DB = PostgresDB;
    fn db(&self) -> &PostgresDB { &self.db }
}

impl HasCache for ProductionScope {
    type Cache = RedisCache;
    fn cache(&self) -> &RedisCache { &self.cache }
}

impl HasLLM for ProductionScope {
    type LLM = ClaudeClient;
    fn llm(&self) -> &ClaudeClient { &self.llm }
}
```

**Compile-time capability checking:**

```rust
// This compiles:
research_agent(&production_scope, "query").await;

// This WON'T compile — missing HasLLM:
struct MinimalScope { db: PostgresDB }
impl HasDatabase for MinimalScope { ... }

research_agent(&minimal_scope, "query").await;
// Error: MinimalScope doesn't implement HasLLM
```

---

### 4. Scope Composition via Trait Bounds

Coproduct-style scope merging becomes trait composition:

```rust
/// Combine capabilities from multiple sources
struct CombinedScope<A, B> {
    a: A,
    b: B,
}

/// Delegate to inner scopes
impl<A: HasDatabase, B> HasDatabase for CombinedScope<A, B> {
    type DB = A::DB;
    fn db(&self) -> &Self::DB { self.a.db() }
}

impl<A, B: HasCache> HasCache for CombinedScope<A, B> {
    type Cache = B::Cache;
    fn cache(&self) -> &Self::Cache { self.b.cache() }
}

// Usage:
let db_scope = DatabaseScope::new(postgres);
let cache_scope = CacheScope::new(redis);
let combined = CombinedScope { a: db_scope, b: cache_scope };

// combined: HasDatabase + HasCache
```

---

### 5. Lifetime-Scoped Agents (No GC/Sandbox)

Rust's borrow checker enforces scope boundaries:

```rust
/// Agent borrows scope — cannot outlive it
struct Agent<'scope, S> {
    scope: &'scope S,
    config: AgentConfig,
}

impl<'scope, S> Agent<'scope, S>
where
    S: HasDatabase + HasLLM,
{
    fn new(scope: &'scope S, config: AgentConfig) -> Self {
        Self { scope, config }
    }

    async fn call(&self, task: &str) -> Result<Output, AgentError> {
        // Can access scope.db(), scope.llm()
        // Cannot store references beyond 'scope lifetime
        let data = self.scope.db().query(task).await?;
        self.scope.llm().process(&data).await
    }
}

// Usage:
fn run_agent() {
    let scope = ProductionScope::new();
    let agent = Agent::new(&scope, config);

    let result = agent.call("task").await;

    // agent dropped here
    // scope dropped here
    // No dangling references possible — compiler enforced
}
```

**No sandbox needed:** The borrow checker prevents agents from accessing resources they don't have references to.

---

### 6. Operadic Orchestration via GATs

Type-level arity enforcement using Generic Associated Types:

```rust
/// An operad operation with typed slots
trait Operation {
    /// Number of inputs (compile-time)
    const ARITY: usize;

    /// Type of each input slot
    type Slot<const I: usize>: Computation;

    /// Output type
    type Output;

    /// Execute with all slots filled
    async fn execute<Slots>(&self, slots: Slots) -> Result<Self::Output, AgentError>
    where
        Slots: SlotsTuple<Self>;
}

/// 3-input orchestrator
struct Orchestrator3;

impl Operation for Orchestrator3 {
    const ARITY: usize = 3;

    type Slot<0> = ResearchAgent;
    type Slot<1> = AnalysisAgent;
    type Slot<2> = SummaryAgent;

    type Output = FinalReport;

    async fn execute<Slots>(&self, slots: Slots) -> Result<FinalReport, AgentError>
    where
        Slots: SlotsTuple<Self>,
    {
        let (research, analysis, summary) = slots.into_tuple();
        // Combine results...
    }
}

// Compile-time arity checking:
orchestrator.execute((research, analysis, summary)).await; // ✓ 3 slots
orchestrator.execute((research, analysis)).await;          // ✗ Compile error: expected 3, got 2
```

---

### 7. Zero-Cost Functor Proxying

Proxying is just borrowing — no RPC overhead:

```rust
/// Functor that maps local types to "proxied" views
trait ProxyFunctor {
    type Local;
    type Proxy<'a> where Self::Local: 'a;

    fn proxy(local: &Self::Local) -> Self::Proxy<'_>;
}

/// Identity proxy — zero cost, just a reference
struct IdentityProxy;

impl<T> ProxyFunctor for IdentityProxy {
    type Local = T;
    type Proxy<'a> = &'a T where T: 'a;

    fn proxy(local: &T) -> &T { local }
}

/// Filtered proxy — only exposes certain methods
struct ReadOnlyProxy;

impl ProxyFunctor for ReadOnlyProxy {
    type Local = Database;
    type Proxy<'a> = ReadOnlyView<'a>;

    fn proxy(local: &Database) -> ReadOnlyView<'_> {
        ReadOnlyView { inner: local }
    }
}

struct ReadOnlyView<'a> {
    inner: &'a Database,
}

impl ReadOnlyView<'_> {
    // Only expose read methods
    async fn query(&self, q: &str) -> Results {
        self.inner.query(q).await
    }
    // write() not exposed — agent can't call it
}
```

---

### 8. Compile-Time Parallel Safety

`Send + Sync` bounds guarantee safe concurrency:

```rust
/// Tool that only needs shared read access
async fn read_tool<S: HasDatabase + Sync>(scope: &S) -> Data {
    scope.db().read().await
}

/// Tool that needs mutable access
async fn write_tool<S: HasDatabase>(scope: &mut S) -> Data {
    scope.db().write().await
}

/// Parallel execution — compiler enforces safety
async fn parallel_agents<S>(scope: Arc<S>) -> (A, B)
where
    S: HasDatabase + HasLLM + Send + Sync,
{
    // Arc<S> is Send — can be moved to other tasks
    // S: Sync — &S can be shared across threads
    let (a, b) = tokio::join!(
        tokio::spawn({
            let s = scope.clone();
            async move { read_tool(&*s).await }
        }),
        tokio::spawn({
            let s = scope.clone();
            async move { read_tool(&*s).await }
        }),
    );
    (a.unwrap(), b.unwrap())
}

// This WON'T compile — can't share &mut across tasks:
async fn bad_parallel<S: HasDatabase>(scope: &mut S) {
    tokio::join!(
        write_tool(scope),  // &mut S
        write_tool(scope),  // &mut S — ERROR: can't borrow mutably twice
    );
}
```

---

### 9. Trace Recording Without Runtime Cost

Use const generics to optionally include tracing:

```rust
/// Computation wrapper that records traces
struct Traced<C, const ENABLED: bool> {
    inner: C,
}

impl<C: Computation> Computation for Traced<C, false> {
    type Input = C::Input;
    type Output = C::Output;

    async fn run(&self, input: Self::Input) -> Result<Self::Output, AgentError> {
        // No tracing — zero overhead
        self.inner.run(input).await
    }
}

impl<C: Computation> Computation for Traced<C, true> {
    type Input = C::Input;
    type Output = (C::Output, TraceNode);

    async fn run(&self, input: Self::Input) -> Result<Self::Output, AgentError> {
        let start = Instant::now();
        let result = self.inner.run(input).await?;
        let trace = TraceNode {
            duration: start.elapsed(),
            type_name: std::any::type_name::<C>(),
        };
        Ok((result, trace))
    }
}

// Production: zero tracing overhead
type ProdPipeline = Traced<MyPipeline, false>;

// Debug: full tracing
type DebugPipeline = Traced<MyPipeline, true>;
```

---

## Comparison: Python Agentica vs Rust-Native

| Aspect | Agentica (Python) | Rust-Native |
|--------|-------------------|-------------|
| Type safety | Runtime validation | Compile-time |
| Scope | `Dict[str, Any]` | Trait bounds |
| Isolation | WASM sandbox | Borrow checker |
| Parallelism | `asyncio.gather` (hope) | `Send + Sync` (proven) |
| Proxying | RPC + serialization | Zero-cost borrows |
| Arity checks | Runtime errors | Compile errors |
| Traces | Always-on overhead | Const generic toggle |
| Composition | Runtime dict merge | Trait composition |

---

## Implementation Roadmap

### Phase 1: Core Abstractions (Sessions 1-6)
- [x] `Shape` with runtime checking
- [ ] `Computation` trait
- [ ] `Then` / `Tensor` composition
- [ ] Trait-based capabilities

### Phase 2: Type-Level Guarantees (Sessions 7-10)
- [ ] Const generic shapes
- [ ] Compile-time dimension checking
- [ ] Zero-cost tracing

### Phase 3: Agent Infrastructure (Sessions 17-19)
- [ ] Operadic orchestration with GATs
- [ ] Lifetime-scoped agents
- [ ] Parallel safety via `Send + Sync`

### Phase 4: Integration (Session 20)
- [ ] LLM client integration
- [ ] Tool registry
- [ ] Demo agents

---

## Key Insight

Agentica's architecture fights Python to achieve safety guarantees that Rust provides automatically.

**The Rust version isn't "Agentica ported to Rust" — it's a fundamentally different design where the type system IS the categorical structure.**

- Composition is type-level `Then<A, B>`
- Capabilities are trait bounds
- Scope isolation is ownership
- Parallel safety is `Send + Sync`

The course builds the foundations. This document shows where they lead.
