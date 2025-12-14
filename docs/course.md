Yep — here's the **21-session Rust capstone plan**, where **each session = one commit**, and each session includes:

* **Lesson** (what you learn)
* **Reading** (short + specific)
* **Build** (exact Rust tasks + files)
* **Tests / Done-when**

I'm going to assume the workspace layout we already picked:

```
compositional-ai-lab-rs/
  crates/{core,diff,prob,nlp,games,operads}/
  demos/
```

---

## Course Map & Dependencies

```
Session 1-2: Foundation (shapes, errors)
     ↓
Session 3: Category theory core (FiniteCategory, Functor)
     ↓
Session 3.5 (Optional): Coproducts + Yoneda (scope merging, dynamic discovery)
     ↓
Session 3.6: Yoneda-Style Capabilities (extensible scope) ← NEW
     ↓
Session 4-5: Diagrams + monoidal composition
     ↓
Session 6: Rendering + Zero-Cost Tracing ← ENHANCED
     ↓
     ├─────────────────┬─────────────────┬────────────┐
     ↓                 ↓                 ↓            │
Sessions 7-10      Sessions 11-14    Sessions 15-16  │
(Autodiff +        (Probability)     (NLP/DisCoCat)  │
 Const Generics)                                     │
     ↓                 ↓                 ↓           │
     └─────────────────┴─────────────────┘           │
                       ↓                             │
                Sessions 17-18 (Games)               │
                       ↓                             │
                Session 18.5: Lifetime-Scoped Agents ← NEW
                       ↓                             │
                Session 19 (Operads) ←───────────────┘
                       ↓
                Session 20 (Integration)
                       ↓
          ┌────────────────────────────┐
          │  SESSION 21: CAPSTONE      │
          │  Rust-Native Agent         │
          │  Framework                 │
          │  (Everything converges!)   │
          └────────────────────────────┘
```

**Rust-Native Sessions (from docs/):**
- **Session 3.6:** Yoneda-style capabilities — extensible scope via Request/Response pattern
- **Session 6:** Zero-cost tracing via const generics (debug vs release)
- **Session 7:** Const generic shapes for compile-time dimension checking
- **Session 18.5:** Lifetime-scoped agents + `Send + Sync` parallel safety

**Checkpoints:**
- ✓ **Checkpoint A (Session 6):** Core diagram system complete — you can build, compose, and visualize any diagram
- ✓ **Checkpoint B (Session 10):** Autodiff working — train a model with gradient descent
- ✓ **Checkpoint C (Session 14):** Probability track complete — causal inference operational
- ✓ **Checkpoint D (Session 18.5):** Games track complete — mechanism design + parallel-safe agents
- ✓ **Checkpoint E (Session 21):** **CAPSTONE** — Full Rust-native agent framework with compile-time safety

---

## Session 1 — Categories as "composition interfaces"

**Reading**

* Leinster, *Basic Category Theory*: Ch 1 (categories)
* (Optional) Milewski, *Category Theory for Programmers*: “Categories Great and Small”

**Lesson**

* Objects + morphisms + associative composition + identities
* Why “composition-first” matches ML pipelines and systems design

**Build (commit 1) — init workspace + core surface**

* Files:

  * root `Cargo.toml` workspace
  * `crates/core/src/lib.rs` with `pub mod shape; pub mod diagram; pub mod error; pub mod cat;`
  * `crates/core/tests/smoke.rs`
* Done-when: `cargo test` passes

---

## Session 2 — Types / shapes as your "objects"

**Reading**

* Fong & Spivak, *Seven Sketches*: the idea of "types as interfaces" (skim)
* (Optional) Leinster: universal properties intro sections
* Rust `typenum` crate docs (overview only)

**Lesson**

* Treat tensor shapes as types that must compose
* Errors as first-class: "non-composable morphisms"
* **Type-level vs runtime shape checking:** understand the tradeoff
  * Runtime `Vec<usize>` — flexible, works with dynamic graphs
  * Compile-time (const generics / `typenum`) — catches errors at compile time, but less flexible
  * We use runtime for flexibility, but add a `StaticShape` trait to show the alternative

**Build (commit 2) — shapes + errors**

* Add `crates/core/src/shape.rs`, `crates/core/src/error.rs`
* Implement:

  * `TypeId(&'static str)` + `Shape { ty: TypeId, dims: Vec<usize> }`
  * constructors: `scalar`, `vector`, `matrix`
  * `CoreError::ShapeMismatch { expected, got }`
  * **Type-level shapes (demonstration):**
    ```rust
    /// Marker trait for compile-time shape checking (optional path)
    pub trait StaticShape {
        const DIMS: &'static [usize];
        fn to_shape() -> Shape;
    }

    /// Example: a 3x3 matrix type known at compile time
    pub struct Mat3x3;
    impl StaticShape for Mat3x3 {
        const DIMS: &'static [usize] = &[3, 3];
        fn to_shape() -> Shape { Shape::matrix(3, 3) }
    }
    ```
* Tests: `core/tests/shape.rs` (constructors + mismatch + static shape conversion)

---

## Session 3 — Functors + naturality (enough to recognize it later)

**Reading**

* Leinster: Ch 2–3 (functors + natural transformations)

**Lesson**

* Functor = structure-preserving translation
* Natural transformation = "same mapping, uniformly across contexts"
* **Why this matters for the rest of the course:**
  * Session 9 (Autodiff): The backward pass is a functor to the *opposite category* — reversing arrows
  * Session 15 (DisCoCat): The semantics functor maps grammar to vector spaces
  * Session 19 (Operads): Operad algebras are functors from the operad to your domain
* We build the infrastructure now; we'll reference it explicitly later

**Build (commit 3) — tiny finite categories**

* Add `crates/core/src/cat.rs`
* Implement:

  * `FiniteCategory` (toy) + `compose` lookup
  * `Functor<C, D>` trait with `map_obj`, `map_mor`, and `check_preserves_id_comp()`
  * `OppositeCategory` wrapper that reverses arrows (needed for Session 9)
  * **Forward reference markers:**
    ```rust
    /// Marker: This will be used in Session 9 to show backprop as a functor
    /// from the forward computation category to its opposite.
    pub struct BackpropFunctorMarker;

    /// Marker: This will be used in Session 15 for grammar → Vect semantics
    pub struct SemanticsFunctorMarker;
    ```
* Tests: `core/tests/cat.rs`
  * tiny category + functor check
  * opposite category reverses composition order
  * identity functor preserves structure

---

## Session 3.5 (Optional) — Coproducts and the Yoneda Perspective

**Reading**

* Leinster: Coproducts / colimits sections
* Milewski: Yoneda lemma chapter (intuition only — skip the proofs)

**Lesson**

* **Coproducts = "merging" objects**
  * If you have objects A and B, their coproduct A + B is the "smallest" object containing both
  * Dual to products: instead of projections, you have injections
  * **Agent application:** Scope merging in systems like Agentica
    ```
    global_scope = {db, cache}
    local_scope  = {api_client}
    merged_scope = global_scope ∪ local_scope  // coproduct!
    ```

* **Yoneda perspective: objects are determined by their morphisms**
  * An object X is fully characterized by all morphisms into it: Hom(-, X)
  * You don't need to "see inside" X — just know what maps to/from it
  * **Agent application:** Dynamic tool discovery
    * Agents discover what an object "is" by calling its methods
    * No explicit schema needed — introspect the interface
    ```python
    # Agent doesn't need a tool definition for `db`
    # It discovers: db.query(), db.insert(), db.close()
    # The object IS its interface (Yoneda intuition)
    ```

* **Why this matters for AI systems:**
  * Scope composition in multi-agent orchestration
  * Dynamic capability discovery without rigid schemas
  * Type-safe merging of resources from different sources

**Build (commit 3.5) — coproducts + representables**

* Add to `crates/core/src/cat.rs`:

  ```rust
  /// Coproduct of two objects in a category (if it exists)
  pub struct Coproduct<A, B> {
      pub sum: String,           // the coproduct object A + B
      pub inj_left: String,      // injection A → A + B
      pub inj_right: String,     // injection B → A + B
  }

  /// Scope: a collection of named objects (like agent scope)
  #[derive(Clone, Debug)]
  pub struct Scope {
      objects: HashMap<String, TypeId>,
  }

  impl Scope {
      pub fn new() -> Self { ... }
      pub fn insert(&mut self, name: &str, ty: TypeId) { ... }

      /// Merge two scopes (coproduct-style)
      pub fn merge(&self, other: &Scope) -> Self {
          let mut merged = self.clone();
          for (k, v) in &other.objects {
              merged.objects.insert(k.clone(), v.clone());
          }
          merged
      }

      /// List available "morphisms" (methods) — Yoneda-style discovery
      pub fn available_methods(&self) -> Vec<String> {
          self.objects.keys().cloned().collect()
      }
  }
  ```

* Tests:
  * Scope merge combines all entries
  * Merge is associative: (A ∪ B) ∪ C = A ∪ (B ∪ C)
  * Empty scope is identity: A ∪ {} = A

---

## Session 3.6 — Yoneda-Style Capabilities (Extensible Scope)

**Reading**

* Leinster, *Basic Category Theory*: Yoneda lemma sections (Ch 4)
* Review `docs/rust-native-agents.md` sections on trait-based capabilities
* Rust book: Trait objects, `Any`, and downcasting

**Lesson**

* **The problem with hardcoded traits:**
  ```rust
  // Must enumerate ALL possible capabilities upfront
  trait HasDatabase { ... }
  trait HasCache { ... }
  trait HasLLM { ... }
  // What about HasVectorDB? HasMetrics? HasAuth? → Endless proliferation
  ```

* **Yoneda's insight:** An object is fully characterized by the morphisms into it.
  Applied to capabilities: **A capability is defined by what operations it supports, not by a name.**

* **The Yoneda-style pattern:**
  ```rust
  // Instead of naming capabilities, define them by their REQUEST/RESPONSE behavior

  /// A request defines what you want and what you get back
  trait Request: Send + 'static {
      type Response: Send + 'static;
  }

  /// A capability can handle certain request types
  trait Capability: Send + Sync {
      fn type_id(&self) -> std::any::TypeId;
  }

  /// The key trait: "this capability handles requests of type R"
  trait Handles<R: Request>: Capability {
      fn handle(&self, req: R) -> Result<R::Response, CapabilityError>;
  }
  ```

* **Why this is Yoneda:**
  * Yoneda: Hom(−, X) determines X (all morphisms INTO X)
  * Here: A capability is determined by all Request types it can Handle
  * You don't ask "is this a Database?" — you ask "can this handle QueryRequest?"

* **Extensibility without modification:**
  ```rust
  // Anyone can define new request types — no changes to core!

  struct SqlQuery { sql: String }
  impl Request for SqlQuery {
      type Response = Vec<Row>;
  }

  struct CacheGet { key: String }
  impl Request for CacheGet {
      type Response = Option<String>;
  }

  struct LlmComplete { prompt: String, max_tokens: usize }
  impl Request for LlmComplete {
      type Response = String;
  }

  // New capability? Just define the request type:
  struct VectorSearch { embedding: Vec<f32>, top_k: usize }
  impl Request for VectorSearch {
      type Response = Vec<(DocId, f32)>;  // (id, similarity)
  }
  // No HasVectorDB trait needed!
  ```

* **The Scope as a capability registry:**
  ```rust
  struct Scope {
      capabilities: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
  }

  impl Scope {
      /// Register a capability
      fn register<C: Capability + 'static>(&mut self, cap: C) {
          self.capabilities.insert(TypeId::of::<C>(), Box::new(cap));
      }

      /// Yoneda-style discovery: what can handle this request?
      fn dispatch<R: Request>(&self, req: R) -> Result<R::Response, CapabilityError>
      where
          // Find capability that handles R
      { ... }
  }
  ```

* **Comparison of approaches:**

  | Approach | Compile-time safety | Extensibility | Use when |
  |----------|---------------------|---------------|----------|
  | `Scope` (3.5) | ✗ None | ✓ Fully dynamic | Prototyping, scripts |
  | Hardcoded traits | ✓ Full | ✗ Must modify code | Closed set of capabilities |
  | **Yoneda-style** | ✓ Response types | ✓ Open extension | Production systems |

**Build (commit 3.6) — Yoneda-style capabilities**

* Add `crates/core/src/capability.rs`
* Implement:

  ```rust
  use std::any::{Any, TypeId};
  use std::collections::HashMap;
  use std::marker::PhantomData;

  /// Error when a capability can't handle a request
  #[derive(Debug, Clone)]
  pub enum CapabilityError {
      NotFound { request_type: &'static str },
      HandlerFailed { message: String },
  }

  /// A request defines an operation and its response type.
  /// This is the "morphism" in the Yoneda sense.
  pub trait Request: Send + 'static {
      type Response: Send + 'static;

      /// Human-readable name for error messages
      fn name() -> &'static str;
  }

  /// Marker trait for capabilities (objects that handle requests)
  pub trait Capability: Send + Sync + 'static {
      fn capability_name(&self) -> &'static str;
  }

  /// A capability that can handle requests of type R.
  /// "Handles<R>" = "has a morphism from R into this capability"
  pub trait Handles<R: Request>: Capability {
      fn handle(&self, req: R) -> Result<R::Response, CapabilityError>;
  }

  // ============================================================
  // Example Request Types (extensible by users)
  // ============================================================

  /// Database query request
  pub struct SqlQuery(pub String);
  impl Request for SqlQuery {
      type Response = Vec<String>;  // Simplified: rows as strings
      fn name() -> &'static str { "SqlQuery" }
  }

  /// Cache get request
  pub struct CacheGet(pub String);
  impl Request for CacheGet {
      type Response = Option<String>;
      fn name() -> &'static str { "CacheGet" }
  }

  /// Cache set request
  pub struct CacheSet { pub key: String, pub value: String }
  impl Request for CacheSet {
      type Response = ();
      fn name() -> &'static str { "CacheSet" }
  }

  /// LLM completion request
  pub struct LlmComplete { pub prompt: String, pub max_tokens: usize }
  impl Request for LlmComplete {
      type Response = String;
      fn name() -> &'static str { "LlmComplete" }
  }

  // ============================================================
  // Capability Registry (Yoneda-style Scope)
  // ============================================================

  /// Type-erased handler for dynamic dispatch
  trait AnyHandler: Send + Sync {
      fn handle_any(&self, req: Box<dyn Any + Send>) -> Result<Box<dyn Any + Send>, CapabilityError>;
  }

  /// Wrapper to make Handles<R> into AnyHandler
  struct HandlerWrapper<C, R> {
      capability: C,
      _phantom: PhantomData<R>,
  }

  impl<C, R> AnyHandler for HandlerWrapper<C, R>
  where
      C: Handles<R>,
      R: Request,
  {
      fn handle_any(&self, req: Box<dyn Any + Send>) -> Result<Box<dyn Any + Send>, CapabilityError> {
          let req = req.downcast::<R>()
              .map_err(|_| CapabilityError::HandlerFailed {
                  message: "Request type mismatch".into()
              })?;
          let response = self.capability.handle(*req)?;
          Ok(Box::new(response))
      }
  }

  /// The capability registry — stores handlers by request TypeId
  #[derive(Default)]
  pub struct CapabilityScope {
      handlers: HashMap<TypeId, Box<dyn AnyHandler>>,
  }

  impl CapabilityScope {
      pub fn new() -> Self {
          Self { handlers: HashMap::new() }
      }

      /// Register a capability for a specific request type
      pub fn register<C, R>(&mut self, capability: C)
      where
          C: Handles<R> + Clone + 'static,
          R: Request,
      {
          let wrapper = HandlerWrapper {
              capability,
              _phantom: PhantomData::<R>,
          };
          self.handlers.insert(TypeId::of::<R>(), Box::new(wrapper));
      }

      /// Check if a request type can be handled (Yoneda: "is there a morphism?")
      pub fn can_handle<R: Request>(&self) -> bool {
          self.handlers.contains_key(&TypeId::of::<R>())
      }

      /// Dispatch a request to its handler
      pub fn dispatch<R: Request>(&self, req: R) -> Result<R::Response, CapabilityError> {
          let handler = self.handlers
              .get(&TypeId::of::<R>())
              .ok_or(CapabilityError::NotFound { request_type: R::name() })?;

          let response = handler.handle_any(Box::new(req))?;

          response.downcast::<R::Response>()
              .map(|b| *b)
              .map_err(|_| CapabilityError::HandlerFailed {
                  message: "Response type mismatch".into()
              })
      }

      /// Merge two scopes (coproduct-style, other wins on conflict)
      pub fn merge(mut self, other: Self) -> Self {
          for (k, v) in other.handlers {
              self.handlers.insert(k, v);
          }
          self
      }
  }
  ```

* Tests:
  * `dispatch` returns correct response type for registered capability
  * `dispatch` returns `NotFound` for unregistered request type
  * `can_handle` correctly reflects registered capabilities
  * New request types work without modifying core (extensibility test)
  * `merge` combines handlers from both scopes
  * Handlers are `Send + Sync` (async-safe)

---

## Session 4 — Open diagrams (string diagram data model)

**Reading**

* Fong & Spivak, *Seven Sketches*: string diagrams / monoidal categories sections

**Lesson**

* A “diagram” is a program: boxes + wires
* Inputs/outputs are boundaries; composition = wiring

**Build (commit 4) — Diagram core + validation**

* Add `crates/core/src/diagram.rs`
* Strongly recommended structure (generic over op payload):

  * `Port { shape: Shape }`
  * `Node<O> { op: O, inputs: Vec<Port>, outputs: Vec<Port> }`
  * `Edge { from_port, to_port }`
  * `Diagram<O> { g: petgraph::DiGraph<Node<O>, Edge>, inputs: Vec<(NodeId, usize)>, outputs: Vec<(NodeId, usize)> }`
* Implement:

  * `add_node`, `connect`, `validate()` (shape checks on connect + boundary consistency)
* Tests: `core/tests/diagram_validate.rs`

---

## Session 5 — Monoidal product: sequential vs parallel composition

**Reading**

* *Seven Sketches*: monoidal categories + “wiring diagrams”

**Lesson**

* `then` = categorical composition
* `tensor` = monoidal (parallel) composition
* Interchange law intuition (won’t prove, but you’ll feel it)

**Build (commit 5) — `then()` + `tensor()` + Parallel Agents**

* In `core::diagram` implement:

  * `Diagram::then(&self, rhs: &Diagram<O>) -> Result<Diagram<O>, CoreError>`
  * `Diagram::tensor(&self, rhs: &Diagram<O>) -> Diagram<O>`
  * internal `remap_node_ids` helper (copy graphs + reconnect)

* **Parallel Agent Infrastructure** in `core::parallel`:

  ```rust
  use std::future::Future;
  use tokio::task::JoinSet;

  /// A parallel executor that runs multiple agents concurrently
  /// This is the runtime interpretation of the tensor product (⊗)
  pub struct ParallelAgents<A> {
      agents: Vec<A>,
  }

  impl<A> ParallelAgents<A> {
      pub fn new() -> Self {
          Self { agents: Vec::new() }
      }

      pub fn with(mut self, agent: A) -> Self {
          self.agents.push(agent);
          self
      }

      /// Tensor two parallel groups: (a₁ ⊗ a₂) ⊗ (b₁ ⊗ b₂) = a₁ ⊗ a₂ ⊗ b₁ ⊗ b₂
      pub fn tensor(mut self, mut other: Self) -> Self {
          self.agents.append(&mut other.agents);
          self
      }
  }

  impl<A, I, O, E> ParallelAgents<A>
  where
      A: Agent<Input = I, Output = O, Error = E> + Send + 'static,
      I: Clone + Send + 'static,
      O: Send + 'static,
      E: Send + 'static,
  {
      /// Execute all agents in parallel, collecting results
      /// This is the semantic interpretation of f ⊗ g
      pub async fn run_all(&self, inputs: Vec<I>) -> Result<Vec<O>, E> {
          let mut set = JoinSet::new();

          for (agent, input) in self.agents.iter().zip(inputs) {
              let agent = agent.clone();
              set.spawn(async move { agent.run(input).await });
          }

          let mut results = Vec::with_capacity(self.agents.len());
          while let Some(result) = set.join_next().await {
              results.push(result.unwrap()?);
          }
          Ok(results)
      }

      /// Fan-out: same input to all agents (broadcast)
      pub async fn fan_out(&self, input: I) -> Result<Vec<O>, E> {
          let inputs = vec![input; self.agents.len()];
          self.run_all(inputs).await
      }
  }

  /// The Agent trait — minimal interface for parallel execution
  pub trait Agent: Clone + Send + Sync {
      type Input: Send;
      type Output: Send;
      type Error: Send;

      fn run(&self, input: Self::Input) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send;
  }
  ```

* **Fan-in Combiner** for merging parallel results:

  ```rust
  /// Combines results from parallel agents
  /// This completes the parallel pattern: fan-out → process → fan-in
  pub trait Combiner<T> {
      type Output;
      fn combine(&self, results: Vec<T>) -> Self::Output;
  }

  /// Simple concatenation combiner
  pub struct Concat;
  impl Combiner<String> for Concat {
      type Output = String;
      fn combine(&self, results: Vec<String>) -> String {
          results.join("\n")
      }
  }

  /// Voting combiner (majority wins)
  pub struct Vote;
  impl<T: Eq + std::hash::Hash + Clone> Combiner<T> for Vote {
      type Output = Option<T>;
      fn combine(&self, results: Vec<T>) -> Option<T> {
          use std::collections::HashMap;
          let mut counts: HashMap<T, usize> = HashMap::new();
          for r in results {
              *counts.entry(r).or_insert(0) += 1;
          }
          counts.into_iter().max_by_key(|(_, c)| *c).map(|(v, _)| v)
      }
  }
  ```

* **Example: Research + Analyze → Combine pipeline**

  ```rust
  // Two parallel research agents
  let researchers = ParallelAgents::new()
      .with(WebSearchAgent::new("technical"))
      .with(WebSearchAgent::new("academic"));

  // Fan-out query to both, collect results
  let findings = researchers.fan_out(query).await?;

  // Combine with a synthesis agent (this is sequential composition)
  let synthesis = SynthesisAgent::new();
  let report = synthesis.run(findings).await?;

  // As a diagram:
  //
  //              ┌─────────────────┐
  //   query ─────│ WebSearch(tech) │─────┐
  //              └─────────────────┘     │    ┌───────────┐
  //                                      ├────│ Synthesis │──── report
  //              ┌─────────────────┐     │    └───────────┘
  //   query ─────│ WebSearch(acad) │─────┘
  //              └─────────────────┘
  //
  //   (search_tech ⊗ search_acad) ; synthesize
  ```

* Tests:

  * compose matching boundaries passes
  * mismatched boundary shapes errors
  * tensor doubles boundary count, preserves shapes
  * parallel agents execute concurrently (timing test)
  * fan-out broadcasts input correctly
  * combiners merge results as expected

---

## Session 6 — String diagram UX: render + DOT export

**Reading**

* *Seven Sketches*: string diagram reasoning (skim)
* (Optional) A quick Graphviz DOT primer

**Lesson**

* Why diagram rendering matters: you debug by *seeing*
* Rewrite mindset: refactors preserve meaning
* **Zero-cost tracing (Rust-native advantage):**
  * Python Agentica: tracing always enabled, always costs tokens/time
  * Rust: const generic toggle — zero cost when disabled
  * Same code, different compile-time flags → completely different runtime behavior

**Build (commit 6) — ASCII + DOT + Zero-Cost Tracing**

* Add to `core::diagram`:

  * `render_ascii()` (node list + edges with port indices + shapes)
  * `to_dot()` (Graphviz DOT string)

* Implement **zero-cost tracing wrapper:**

  ```rust
  /// Const generic toggle — zero cost when disabled
  pub struct Traced<C, const ENABLED: bool> {
      inner: C,
  }

  impl<C: Computation> Computation for Traced<C, false> {
      type Input = C::Input;
      type Output = C::Output;

      async fn run(&self, input: Self::Input) -> Result<Self::Output, CoreError> {
          // No tracing code generated — zero overhead
          self.inner.run(input).await
      }
  }

  impl<C: Computation> Computation for Traced<C, true> {
      type Input = C::Input;
      type Output = (C::Output, TraceNode);

      async fn run(&self, input: Self::Input) -> Result<Self::Output, CoreError> {
          let start = std::time::Instant::now();
          let result = self.inner.run(input).await?;
          let trace = TraceNode {
              duration: start.elapsed(),
              type_name: std::any::type_name::<C>(),
          };
          Ok((result, trace))
      }
  }

  /// Compile-time selection via cfg:
  #[cfg(debug_assertions)]
  pub type TracedDiagram<O> = Traced<Diagram<O>, true>;   // Debug: full tracing

  #[cfg(not(debug_assertions))]
  pub type TracedDiagram<O> = Traced<Diagram<O>, false>;  // Release: zero overhead
  ```

* **Key insight:** The same code path gets completely different behavior:
  * `cargo run` → full tracing, detailed logs
  * `cargo run --release` → zero tracing overhead, production-ready

* Add `demos` CLI:

  * `cargo run -p demos -- render --dot` prints DOT for a tiny example
  * `cargo run -p demos -- render --trace` shows trace output (debug only)

* Tests:
  * snapshot-like string contains expected nodes/edges
  * `Traced<_, false>` output type equals inner output type
  * `Traced<_, true>` output type includes `TraceNode`

---

# Differentiation / Backprop track (Sessions 7–10)

## Session 7 — Computation graphs as diagrams

**Reading**

* "Reverse-mode autodiff" overview (any short note)
* Optional: Fong/Spivak/Tuyéras *Backprop as Functor* (intro/abstract)

**Lesson**

* Evaluation is a monoidal/categorical fold over a DAG
* Shapes guarantee composability before runtime
* **Const generic shapes (Rust-native advantage):**
  * Session 2 introduced `StaticShape` as a marker trait
  * Now we implement actual const generic tensors for compile-time dimension checking
  * Trade-off: less flexible than runtime shapes, but catches dimension errors at compile time

**Build (commit 7) — `diff` crate scaffolding + Tensor**

* Add `crates/diff/src/lib.rs`, `tensor.rs`, `ops.rs`
* Implement **runtime `Tensor`** (flexible, dynamic):

  * `Scalar(f32)`, `Vec(Array1<f32>)`, `Mat(Array2<f32>)`
  * `shape()` -> `Shape`

* Implement **const generic `StaticTensor`** (rigid, compile-time checked):

  ```rust
  /// Compile-time dimensioned tensor
  /// Shape errors become compile errors, not runtime panics
  #[derive(Clone, Debug)]
  pub struct StaticTensor<T, const N: usize> {
      data: [T; N],
  }

  /// Type aliases for common shapes
  pub type Scalar<T> = StaticTensor<T, 1>;
  pub type Vector<T, const N: usize> = StaticTensor<T, N>;

  /// A linear layer with compile-time dimension checking
  pub struct Linear<const IN: usize, const OUT: usize> {
      weights: [[f32; IN]; OUT],
      bias: [f32; OUT],
  }

  impl<const IN: usize, const OUT: usize> Linear<IN, OUT> {
      pub fn forward(&self, input: Vector<f32, IN>) -> Vector<f32, OUT> {
          // Shape mismatch is IMPOSSIBLE — won't compile
          todo!()
      }
  }

  // Composing layers — dimensions must chain:
  // This compiles: 784 → 512 → 256 → 10
  type MLP = (Linear<784, 512>, Linear<512, 256>, Linear<256, 10>);

  // This WON'T compile — dimension mismatch:
  // type BadMLP = (Linear<784, 512>, Linear<256, 10>);  // 512 ≠ 256
  ```

* **When to use which:**
  * `Tensor` (runtime): dynamic graphs, variable batch sizes, flexibility
  * `StaticTensor` (const generic): fixed architectures, maximum safety, zero-cost abstractions
* Add `DiffOp` enum with ports:

  * `Add` — elementwise addition, 2 inputs → 1 output (same shape)
  * `ReLU` — elementwise max(0, x), 1 input → 1 output (same shape)
  * `MatMul` — matrix multiplication, 2 inputs → 1 output (shape: [m,k] × [k,n] → [m,n])
  * `SumAll` — reduce to scalar, 1 input → 1 scalar output
  * `Copy` (fan-out/diagonal) — duplicates input to multiple outputs, 1 input → N outputs (same shape each)
    ```rust
    /// Copy is the categorical "diagonal" — it duplicates data.
    /// This is NOT the identity; identity would be 1→1 with no duplication.
    /// Copy(2) means: one input wire splits into two output wires.
    Copy { fan_out: usize },
    ```
* Make `DiffOp` usable as `O` in `Diagram<DiffOp>`
* Tests: op port shapes + tensor shape mapping

---

## Session 8 — Forward evaluation

**Reading**

* Any minimal example of topological evaluation of a DAG

**Lesson**

* Deterministic semantics: diagram + inputs → outputs
* Topological sort as “execution order” (not the math, just the mechanism)

**Build (commit 8) — forward interpreter**

* Add `crates/diff/src/forward.rs`
* Implement:

  * topo-order over `Diagram<DiffOp>`
  * execute each node given input tensors (from incoming edges / boundary inputs)
* Add demo:

  * `demos diff forward` runs a tiny graph and prints output
* Tests:

  * scalar sanity (`(x+y).sum -> …`)
  * matrix matmul sanity

---

## Session 9 — Reverse-mode AD (VJP rules)

**Reading**

* Backprop chain rule derivation (any concise source)
* *Backprop as Functor* sections describing compositionality — **required this session**

**Lesson**

* Reverse-mode = propagate adjoints backward along wires
* Each op implements **VJP**: output-grads → input-grads
* **Connecting to Session 3:** The backward pass is literally a functor!
  * Forward: morphisms go A → B
  * Backward: morphisms go B → A (opposite category)
  * The VJP of a composition is the composition of VJPs in reverse order
  * This is why we built `OppositeCategory` in Session 3

**Build (commit 9) — backward pass**

* Add `crates/diff/src/backward.rs`
* Implement VJP rules:

  * `Add`: grad flows equally to both inputs (∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1)
  * `ReLU`: grad × (input > 0 ? 1 : 0)
  * `MatMul`: ∂L/∂A = (∂L/∂C) × Bᵀ, ∂L/∂B = Aᵀ × (∂L/∂C)
  * `SumAll`: grad broadcasts back to input shape
  * `Copy`: grads from all outputs sum together (adjoint of diagonal is sum)
* Provide `grad(diagram, inputs, seed_grad)` returning input grads
* **Explicit functor connection:**
  ```rust
  /// Demonstrates that backward pass is functorial:
  /// For composed operations f;g, the VJP is vjp(g);vjp(f)
  /// This uses OppositeCategory from Session 3's cat.rs
  pub fn demonstrate_backprop_functor(
      forward_diagram: &Diagram<DiffOp>
  ) -> Diagram<VjpOp> {
      // Returns the "reversed" diagram operating on gradients
  }
  ```
* Tests:

  * numeric grad-check for scalar graph (tolerance: 1e-5)
  * small matmul grad-check (finite differences, h=1e-4)
  * verify `demonstrate_backprop_functor` reverses composition order

---

## Session 10 — Parameters + SGD training loop

**Reading**

* Minimal SGD / training loop refresher

**Lesson**

* Parameter nodes are just inputs you also update
* Training = (forward → loss → backward → update) composed

**Build (commit 10) — Params + tiny training demo**

* Add `DiffOp::Param { id }` or treat params as extra boundary inputs + mapping
* Add `optimizer.rs`: SGD step
* Demo:

  * linear regression on synthetic data (few steps) decreases loss
* Tests:

  * loss decreases after N steps (allow tolerance)

---

# Probability / causality track (Sessions 11–14)

## Session 11 — Stochastic maps (finite Markov kernels)

**Reading**

* Tobias Fritz: Markov categories intro notes (focus on finite kernels)
* Optional: Baez/Fong style "stochastic matrices as morphisms" intuition

**Lesson**

* `X → Dist(Y)` as morphisms
* In finite land: stochastic map = row-stochastic matrix

**Build (commit 11) — `prob` crate + FiniteDist**

* Add `crates/prob/src/lib.rs`, `finite.rs`
* Implement:

  * `FiniteSet { n: usize }`
  * `Dist { p: Array1<f32> }` (normalized)
    ```rust
    const PROB_TOLERANCE: f32 = 1e-6;

    impl Dist {
        pub fn new(p: Array1<f32>) -> Result<Self, ProbError> {
            let sum = p.sum();
            if (sum - 1.0).abs() > PROB_TOLERANCE {
                return Err(ProbError::NotNormalized { sum });
            }
            if p.iter().any(|&x| x < -PROB_TOLERANCE) {
                return Err(ProbError::NegativeProbability);
            }
            Ok(Self { p })
        }

        /// Normalize a vector of weights into a distribution
        pub fn from_weights(weights: Array1<f32>) -> Result<Self, ProbError> {
            let sum = weights.sum();
            if sum <= 0.0 {
                return Err(ProbError::ZeroWeights);
            }
            Self::new(weights / sum)
        }
    }
    ```
  * `Kernel { k: Array2<f32> }` (rows sum to 1 ± PROB_TOLERANCE)
    ```rust
    impl Kernel {
        pub fn new(k: Array2<f32>) -> Result<Self, ProbError> {
            for (i, row) in k.rows().into_iter().enumerate() {
                let sum = row.sum();
                if (sum - 1.0).abs() > PROB_TOLERANCE {
                    return Err(ProbError::RowNotNormalized { row: i, sum });
                }
            }
            Ok(Self { k })
        }
    }
    ```
  * compose kernels via matrix multiplication
* Tests:

  * row sums within tolerance (1e-6)
  * reject distributions that sum to 0.99 or 1.01
  * composition shape correctness
  * composed kernel rows still sum to 1 (within tolerance)

---

## Session 12 — Bayesian networks as composed kernels

**Reading**

* Any short Bayes net factorization overview

**Lesson**

* BN = composing conditionals
* Marginalization = summing out (a linear operation)

**Build (commit 12) — BayesNet + marginal**

* Add `crates/prob/src/bayesnet.rs`
* Implement a tiny BN representation (DAG + CPTs as kernels)
* Provide:

  * sample-free exact joint/marginal for small nets
  * `marginal(var)` via summing over others
* Demo:

  * `demos prob bn` prints marginals
* Tests: known 3-node BN yields expected marginal

---

## Session 13 — Conditioning / inference (exact)

**Reading**

* Bayes rule + evidence updates

**Lesson**

* Conditioning is not plain composition; it’s normalization after multiplying by likelihood
* Importance sampling is optional; exact is fine for finite small

**Build (commit 13) — evidence + query**

* Implement:

  * `condition(var=value)` returning updated distribution / posterior
  * `query(target | evidence)` exact for small BN
* Tests:

  * posterior sums to 1
  * matches hand-computed toy example

---

## Session 14 — Causality: interventions `do(X=x)`

**Reading**

* Pearl do-calculus intro (high level)
* Optional: Fritz notes sections about interventions

**Lesson**

* `do(X=x)` = cut incoming edges into X and replace with constant
* Causal vs observational difference in a confounding example

**Build (commit 14) — intervene**

* Add `intervene(var, value)`:

  * modifies BN: removes parents of var, sets var distribution to delta
* Demo:

  * show `P(Y|X=x)` vs `P(Y|do(X=x))` differs in confounder BN
* Tests: confounder case produces different results

---

## Session 14.5 (Optional) — Continuous distributions and Gaussian kernels

**Reading**

* Any introduction to Gaussian distributions and convolution
* Optional: Fritz notes on continuous Markov kernels

**Lesson**

* Not all probability is finite — continuous distributions are essential for real ML
* Gaussian kernel: `X → N(μ(x), σ²)` where μ depends on input
* **Key insight:** Composition of Gaussian kernels = convolution
  * If K₁: X → N(ax, σ₁²) and K₂: Y → N(by, σ₂²)
  * Then K₂ ∘ K₁: X → N(abx, b²σ₁² + σ₂²)
* Connection to variational inference: approximate posteriors are often Gaussian

**Build (commit 14.5) — continuous kernels**

* Add `crates/prob/src/continuous.rs`
* Implement:

  * `Gaussian { mean: f64, variance: f64 }`
  * `LinearGaussianKernel { slope: f64, intercept: f64, noise_var: f64 }`
    ```rust
    /// A linear Gaussian kernel: x ↦ N(slope * x + intercept, noise_var)
    pub struct LinearGaussianKernel {
        pub slope: f64,
        pub intercept: f64,
        pub noise_var: f64,
    }

    impl LinearGaussianKernel {
        /// Compose two linear Gaussian kernels
        /// K2 ∘ K1 where K1: x → N(a1*x + b1, v1), K2: y → N(a2*y + b2, v2)
        /// Result: x → N(a2*a1*x + a2*b1 + b2, a2²*v1 + v2)
        pub fn compose(&self, other: &Self) -> Self {
            LinearGaussianKernel {
                slope: self.slope * other.slope,
                intercept: self.slope * other.intercept + self.intercept,
                noise_var: self.slope.powi(2) * other.noise_var + self.noise_var,
            }
        }
    }
    ```
  * `sample(kernel, x, rng)` for Monte Carlo when exact inference isn't tractable
* Demo:

  * Chain 3 linear Gaussian kernels, show that composed kernel matches empirical samples
* Tests:

  * Composed kernel parameters match analytical formula
  * Empirical mean/variance from samples within tolerance of analytical values

---

# DisCoCat NLP track (Sessions 15–16)

## Session 15 — Pregroup grammars and type reductions

**Reading**

* Coecke–Sadrzadeh–Clark: DisCoCat overview (intro + examples)
* Lambek: Pregroups intro (the key definitions only)

**Lesson**

* **Pregroup grammar basics:**
  * Types: `N` (noun), `S` (sentence), and adjoints `Nˡ`, `Nʳ`, `Sˡ`, `Sʳ`
  * Reduction rules: `N · Nʳ → 1` and `Nˡ · N → 1` (types "cancel")
  * A sentence is grammatical iff the concatenated types reduce to `S`
* **Example parse: "Alice loves Bob"**
  ```
  Alice : N
  loves : Nʳ · S · Nˡ    (takes noun on left, noun on right, produces sentence)
  Bob   : N

  Concatenated: N · (Nʳ · S · Nˡ) · N
  Reductions:   N · Nʳ → 1, Nˡ · N → 1
  Result:       S ✓
  ```
* Why this matters: reductions tell us which wires to contract in the tensor network

**Build (commit 15) — `nlp` crate + pregroup grammar**

* Add `crates/nlp/src/lib.rs`, `grammar.rs`, `pregroup.rs`
* Implement:

  * `GrammarType` enum: `N`, `S`, `LeftAdj(Box<GrammarType>)`, `RightAdj(Box<GrammarType>)`
  * `Word { text: String, grammar_type: GrammarType }`
  * `reduce(types: &[GrammarType]) -> Option<GrammarType>` — applies reduction rules
  * `is_grammatical(sentence: &[Word]) -> bool` — reduces to `S`
  ```rust
  #[derive(Clone, Debug, PartialEq)]
  pub enum GrammarType {
      N,                              // Noun
      S,                              // Sentence
      LeftAdj(Box<GrammarType>),      // Xˡ (wants X on the right)
      RightAdj(Box<GrammarType>),     // Xʳ (wants X on the left)
  }

  /// Transitive verb type: Nʳ · S · Nˡ
  pub fn transitive_verb() -> Vec<GrammarType> {
      vec![
          GrammarType::RightAdj(Box::new(GrammarType::N)),
          GrammarType::S,
          GrammarType::LeftAdj(Box::new(GrammarType::N)),
      ]
  }
  ```
* Tests:
  * "Alice loves Bob" is grammatical
  * "loves Alice Bob" is not grammatical
  * "Alice Bob loves" is not grammatical (in English word order)

---

## Session 15.5 — Tensor semantics: from grammar to vectors

**Reading**

* DisCoCat paper: sections on functorial semantics
* **Callback to Session 3:** This is the `SemanticsFunctorMarker` in action!

**Lesson**

* **The semantics functor F: Grammar → Vect**
  * F(N) = ℝⁿ (noun space, e.g., n=2 for toy example)
  * F(S) = ℝ (sentence space = scalars, or ℝᵐ for richer semantics)
  * F(Xˡ) = F(Xʳ) = F(X)* ≅ F(X) (dual space, isomorphic for finite-dim)
* **Word meanings as tensors:**
  * Noun "Alice" → vector in ℝⁿ
  * Transitive verb "loves" → tensor in ℝⁿ ⊗ ℝ ⊗ ℝⁿ ≅ ℝⁿˣⁿ (a matrix!)
* **Sentence meaning = tensor contraction following parse**
  * Contract indices that correspond to reduction steps
  * Result: scalar (or sentence vector) representing meaning

**Build (commit 15.5) — tensor semantics**

* Add `crates/nlp/src/semantics.rs`
* Implement:

  * `NounMeaning { vec: Array1<f32> }` — noun as vector
  * `VerbMeaning { tensor: Array2<f32> }` — transitive verb as matrix (simplified)
  * `Lexicon` mapping words to their tensor meanings
  * `contract_sentence(words: &[Word], lexicon: &Lexicon) -> Array1<f32>`
    ```rust
    /// Computes sentence meaning by tensor contraction
    /// For "Alice loves Bob": alice_vec · loves_mat · bob_vec
    /// This is matrix-vector multiplication: (alice · loves) · bob
    pub fn contract_sentence(
        subject: &NounMeaning,
        verb: &VerbMeaning,
        object: &NounMeaning,
    ) -> f32 {
        // subject^T @ verb_matrix @ object
        subject.vec.dot(&verb.tensor.dot(&object.vec))
    }
    ```
* Tests:
  * Contraction produces scalar output
  * Dimension mismatch produces error

---

## Session 16 — DisCoCat demo + similarity

**Reading**

* Same paper: worked examples section
* Optional: Papers on learned DisCoCat embeddings

**Lesson**

* Composition beats bag-of-words when structure matters
* **Concrete example:**
  * "Alice loves Bob" vs "Bob loves Alice" — different meanings!
  * Bag-of-words: identical (same word set)
  * DisCoCat: different (verb matrix not symmetric)
* How this relates to learned embeddings (word2vec, transformers)

**Build (commit 16) — demo**

* `demos nlp discocat`:

  * Define toy lexicon:
    ```rust
    let lexicon = Lexicon::new()
        .add_noun("alice", vec![1.0, 0.0])
        .add_noun("bob", vec![0.0, 1.0])
        .add_noun("charlie", vec![0.7, 0.7])
        .add_verb("loves", array![[0.9, 0.1], [0.2, 0.8]])  // asymmetric!
        .add_verb("sees", array![[0.5, 0.5], [0.5, 0.5]]); // symmetric
    ```
  * Compute embeddings for:
    * "Alice loves Bob"
    * "Bob loves Alice"
    * "Alice sees Bob"
    * "Bob sees Alice"
  * Show: loves-sentences differ, sees-sentences are equal
  * Cosine similarity comparisons between all pairs
* Tests:
  * "Alice loves Bob" ≠ "Bob loves Alice" (asymmetric verb)
  * "Alice sees Bob" ≈ "Bob sees Alice" (symmetric verb)
  * Results stable given fixed RNG seed

---

# Open games / incentives track (Sessions 17–18)

## Session 17 — Open systems: agent ⊗ environment ⊗ objective

**Reading**

* Intro material on “open games” (high level; even blog notes are fine)
* Optional: Hedges et al. open games paper (skim definitions)

**Lesson**

* Model components as composable boxes
* Swap policy / env / objective without rewiring everything

**Build (commit 17) — `games` crate rollout composer**

* Add `crates/games/src/lib.rs`, `policy.rs`, `env.rs`, `rollout.rs`
* Implement:

  * `Policy: Obs -> Act`
  * `Env: (State, Act) -> (State, Obs, Reward)`
  * `rollout(policy, env, T)` returns trajectory
* Demo: `demos games rollout`

---

## Session 18 — Mechanisms as rewires (subnet-native)

**Reading**

* Any mechanism design / scoring rule short primer (or your own notes)
* Optional: Algorithmic Game Theory (Nisan et al.), Chapter 9 on mechanism design

**Lesson**

* Incentives = objective function + measurement channel
* Exploits = degenerate strategies under the objective
* **Concrete game specification:**
  * 2 agents (Alice, Bob), each choosing from 3 actions: {Honest, Exploit, Defect}
  * Environment produces a signal based on both actions
  * Mechanism scores agents based on signal + their reported action

**Build (commit 18) — toy incentive game**

* Implement a **concrete 2-player mechanism game:**

  ```rust
  /// Actions available to each agent
  #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
  pub enum Action {
      Honest,   // Play by the rules
      Exploit,  // Game the measurement system
      Defect,   // Actively harmful
  }

  /// Payoff matrix for Mechanism A (naive scoring)
  /// Format: payoffs[my_action][opponent_action] = (my_score, opponent_score)
  pub const MECHANISM_A: [[i32; 3]; 3] = [
      // vs Honest, vs Exploit, vs Defect
      [3, 0, 1],   // Honest
      [5, 2, 4],   // Exploit  <- dominates Honest!
      [2, 1, 0],   // Defect
  ];

  /// Payoff matrix for Mechanism B (proper scoring rule)
  /// Exploit is no longer dominant
  pub const MECHANISM_B: [[i32; 3]; 3] = [
      // vs Honest, vs Exploit, vs Defect
      [4, 3, 2],   // Honest   <- now best response to Honest
      [2, 1, 3],   // Exploit  <- punished
      [1, 2, 0],   // Defect
  ];

  /// Find Nash equilibria by brute-force best-response iteration
  pub fn find_equilibria(payoffs: &[[i32; 3]; 3]) -> Vec<(Action, Action)> {
      // Check all 9 action pairs for mutual best response
  }
  ```

* Implement:
  * `Mechanism` struct with payoff matrix
  * `best_response(mechanism, opponent_action) -> Action`
  * `find_equilibria(mechanism) -> Vec<(Action, Action)>`
  * `expected_social_welfare(mechanism, strategy_profile) -> f32`

* Demo: `demos games mechanism`
  * Print payoff matrices for both mechanisms
  * Show Nash equilibria for each:
    * Mechanism A: (Exploit, Exploit) is the unique equilibrium
    * Mechanism B: (Honest, Honest) is an equilibrium
  * Compare social welfare at equilibrium

* Tests:
  * Mechanism A: `best_response(*, Honest) == Exploit`
  * Mechanism A: (Exploit, Exploit) is equilibrium
  * Mechanism B: `best_response(*, Honest) == Honest`
  * Mechanism B: (Honest, Honest) is equilibrium
  * Social welfare higher under Mechanism B equilibrium

---

## Session 18.5 — Lifetime-Scoped Agents and Parallel Safety

**Reading**

* Review `docs/rust-native-agents.md` sections on lifetime-scoped agents
* Rust book: Lifetimes, `Send`, `Sync` traits
* Tokio docs: Spawning tasks with `Send` bounds

**Lesson**

* **Why Python needs sandboxes, Rust doesn't:**
  * Python Agentica: WASM sandbox + microVM for isolation (runtime overhead)
  * Rust: Borrow checker enforces isolation at compile time (zero overhead)
  * Agent literally cannot access what's not in its scope — compiler proves it

* **Lifetime-scoped agents:**
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
      let resources = Resources::new();
      {
          let agent = Agent::new(&resources, config);
          agent.call("task").await;
      }  // agent dropped, scope borrow ends

      // resources still valid here, agent gone
      // No leaks, no dangling refs — compiler enforced
  }
  ```

* **Parallel safety via `Send + Sync`:**
  * `Send`: type can be transferred to another thread
  * `Sync`: type can be shared between threads via `&T`
  * Compiler proves parallel safety — data races are compile errors

  ```rust
  /// Parallel execution — compiler enforces safety
  async fn parallel_agents<S>(scope: Arc<S>) -> (A, B)
  where
      S: HasDatabase + HasLLM + Send + Sync,
  {
      // Arc<S> is Send — can be moved to other tasks
      // S: Sync — &S can be shared across threads
      tokio::join!(
          tokio::spawn({
              let s = scope.clone();
              async move { agent1(&*s).await }
          }),
          tokio::spawn({
              let s = scope.clone();
              async move { agent2(&*s).await }
          }),
      )
  }

  // This WON'T compile — can't share &mut across tasks:
  async fn bad_parallel<S: HasDatabase>(scope: &mut S) {
      tokio::join!(
          write_tool(scope),  // &mut S
          write_tool(scope),  // &mut S — ERROR: can't borrow mutably twice
      );
  }
  ```

* **Comparison to Python's `asyncio.gather`:**
  * Python: "hope these don't race" — runtime errors if they do
  * Rust: "compiler proves these don't race" — won't compile if unsafe

**Build (commit 18.5) — lifetime-scoped agents**

* Add `crates/games/src/agent.rs`
* Implement:

  ```rust
  /// A lifetime-scoped agent that borrows its capabilities
  pub struct ScopedAgent<'scope, S> {
      scope: &'scope S,
      name: String,
  }

  impl<'scope, S> ScopedAgent<'scope, S>
  where
      S: Send + Sync,
  {
      pub fn new(scope: &'scope S, name: impl Into<String>) -> Self {
          Self { scope, name: name.into() }
      }
  }

  /// Parallel agent execution with proven safety
  pub async fn run_parallel<S, A, B, FA, FB>(
      scope: Arc<S>,
      agent_a: FA,
      agent_b: FB,
  ) -> (A, B)
  where
      S: Send + Sync + 'static,
      A: Send + 'static,
      B: Send + 'static,
      FA: FnOnce(Arc<S>) -> A + Send + 'static,
      FB: FnOnce(Arc<S>) -> B + Send + 'static,
  {
      let (a, b) = tokio::join!(
          tokio::spawn({
              let s = scope.clone();
              async move { agent_a(s) }
          }),
          tokio::spawn({
              let s = scope.clone();
              async move { agent_b(s) }
          }),
      );
      (a.unwrap(), b.unwrap())
  }
  ```

* Tests:
  * `ScopedAgent` cannot outlive its scope (compile-time check — use `trybuild` crate)
  * Parallel execution with `Send + Sync` scope compiles and runs
  * Parallel execution with non-`Sync` scope fails to compile
  * Multiple read-only borrows in parallel succeed
  * Concurrent mutable borrows fail to compile

---

# Operads + integration (Sessions 19–20)

## Session 19 — Operads: multi-input wiring constraints

**Reading**

* Spivak: wiring diagrams / operad intuition notes (high level)
* Optional: "Operads for complex system design" (any accessible intro)

**Lesson**

* Operad = "how to plug k things into one thing"
* Tool-using agents and pipelines are operadic in practice
* **Why operads vs plain diagrams?**
  * Diagrams: nodes + edges, any wiring allowed if types match
  * Operads: explicit **nesting structure** — which operations can contain which
  * Key difference: operads enforce **arity constraints** and **hierarchical composition**

* **Motivating example: LLM tool-use pipeline**
  ```
  Agent(3) — takes 3 tool outputs, produces 1 response
    ├── Tool("search")  — 1 query → 1 result
    ├── Tool("calculate") — 2 numbers → 1 result
    └── Tool("fetch") — 1 url → 1 result

  Valid wiring: Agent receives (search_result, calc_result, fetch_result)
  Invalid wiring: Agent receives (search_result, search_result) — wrong arity!
  ```
  * With plain `Diagram`: you could accidentally wire 2 inputs instead of 3
  * With `Operad`: the operation signature `Agent(3)` enforces exactly 3 inputs

* **What operads catch that diagrams don't:**
  * Arity mismatches (wrong number of inputs)
  * Nesting violations (operation A can't be inside operation B)
  * Scope constraints (variable X only visible within block Y)

**Build (commit 19) — `operads` crate**

* Add `crates/operads/src/lib.rs`, `wiring.rs`
* Implement:

  ```rust
  /// An operation with explicit arity
  pub struct Operation {
      pub name: String,
      pub arity: usize,  // number of inputs expected
      pub inputs: Vec<Shape>,
      pub outputs: Vec<Shape>,
  }

  /// A wiring plan that enforces operadic constraints
  pub struct WiringPlan {
      pub outer: Operation,
      pub inner: Vec<Operation>,
      pub wiring: Vec<(usize, usize)>,  // (inner_op_idx, outer_slot_idx)
  }

  impl WiringPlan {
      /// Validates that:
      /// 1. Number of inner operations == outer.arity
      /// 2. Each inner output shape matches expected outer input shape
      /// 3. No slot is wired twice
      pub fn validate(&self) -> Result<(), OperadError> { ... }

      /// Convert to a core::Diagram (loses arity enforcement)
      pub fn to_diagram(&self) -> Diagram<Operation> { ... }
  }
  ```

* **Demo showing the difference:**
  ```rust
  // This should FAIL with operads but SUCCEED with plain diagrams
  let agent = Operation::new("Agent", 3, ...);  // expects 3 inputs
  let search = Operation::new("Search", 1, ...);
  let calc = Operation::new("Calc", 2, ...);

  // Wrong: only 2 inner operations for arity-3 outer
  let bad_plan = WiringPlan {
      outer: agent,
      inner: vec![search, calc],  // only 2!
      wiring: vec![(0, 0), (1, 1)],
  };
  assert!(bad_plan.validate().is_err());  // Operad catches this!
  ```

* Tests:
  * Correct arity wiring succeeds
  * Wrong arity (too few inputs) fails with `OperadError::ArityMismatch`
  * Wrong arity (too many inputs) fails
  * Shape mismatch fails with `OperadError::ShapeMismatch`
  * Valid plan converts to equivalent `Diagram`

---

## Session 20 — Polish + Integration (Preparing for Capstone)

**Reading**

* Re-skim *Seven Sketches* intro: "composition is the interface"
* Review `docs/rust-native-agents.md` — what's coming in Session 21

**Lesson**

* Unification: same core wiring abstraction supports learning/probability/semantics/games
* Present it as a systems tool, not "math homework"
* **Preparing for the capstone:** Review how each component will contribute:
  * Shapes → Tool input/output types
  * Diagrams → Agent execution traces
  * Tracing → Zero-cost debug logs
  * Capabilities → Compile-time resource requirements

**Build (commit 20) — integration**

* `demos` provides subcommands:

  * `render`, `diff`, `prob`, `nlp`, `games`, `operads`
* Add `README.md`:

  * 1 page narrative + how to run each demo
  * include ASCII render snippets
* Add workspace-wide `cargo test` + clippy cleanups
* **Pre-capstone checklist:**
  * [ ] All crates compile with `--all-features`
  * [ ] `compositional_core::Shape` exports cleanly
  * [ ] `compositional_core::Diagram` is generic and reusable
  * [ ] Tracing wrapper compiles with const generics
  * [ ] All tests pass: `cargo test --workspace`
* Done-when: `cargo run -p demos -- diff` etc all work, ready for Session 21

---

## Session 20.5 (Optional) — Backend abstraction and GPU targeting

**Reading**

* Overview of compute backends (CPU, CUDA, WebGPU)
* Optional: WGPU Rust crate documentation (skim)

**Lesson**

* **Why backend abstraction matters:**
  * The `Diagram` representation is backend-agnostic
  * Same diagram can execute on CPU (ndarray), GPU (CUDA/WebGPU), or distributed systems
  * This is the payoff of compositional design: swap backends without changing logic

* **The compilation pipeline:**
  ```
  Diagram<DiffOp>
       │
       ▼
  ┌─────────────────┐
  │  Backend Trait  │
  │  - compile()    │
  │  - execute()    │
  └─────────────────┘
       │
       ├──► CpuBackend (ndarray)
       ├──► CudaBackend (future)
       └──► WgpuBackend (future)
  ```

* **Key insight:** The diagram's structure enables optimizations:
  * Fusion: combine adjacent operations into single kernels
  * Parallelization: identify independent subgraphs
  * Memory planning: pre-allocate buffers based on shape analysis

**Build (commit 20.5) — backend trait**

* Add `crates/diff/src/backend.rs`
* Implement:

  ```rust
  /// Backend trait for executing diagrams
  pub trait Backend {
      type Buffer;  // Backend-specific tensor storage

      /// Compile a diagram into an executable form
      fn compile(&self, diagram: &Diagram<DiffOp>) -> CompiledGraph<Self::Buffer>;

      /// Execute the compiled graph with inputs
      fn execute(
          &self,
          graph: &CompiledGraph<Self::Buffer>,
          inputs: &[Self::Buffer],
      ) -> Vec<Self::Buffer>;
  }

  /// CPU backend using ndarray (already implemented in Sessions 7-10)
  pub struct CpuBackend;

  impl Backend for CpuBackend {
      type Buffer = Tensor;  // Our existing Tensor type

      fn compile(&self, diagram: &Diagram<DiffOp>) -> CompiledGraph<Self::Buffer> {
          // Topological sort + operation scheduling
      }

      fn execute(&self, graph: &CompiledGraph<Self::Buffer>, inputs: &[Self::Buffer]) -> Vec<Self::Buffer> {
          // Same as forward.rs, but using Backend abstraction
      }
  }

  /// Placeholder for future GPU backend
  pub struct GpuBackend {
      // Would hold WGPU device, queue, etc.
  }
  ```

* **Conceptual demo** (no actual GPU code required):
  * Show how the same `Diagram<DiffOp>` can be:
    1. Executed directly with `CpuBackend`
    2. Serialized to a "kernel plan" that could target GPU
  * Print the kernel plan showing what a GPU compiler would need

* Tests:
  * `CpuBackend` produces same results as direct `forward()` execution
  * Compiled graph caches correctly (re-execution doesn't recompile)
  * Backend trait is object-safe (can use `dyn Backend`)

---

# Capstone: Rust-Native Agent Framework (Session 21)

## Session 21 — Building a Complete Agent System

*This is the capstone project. Everything you've built culminates here.*

**Reading**

* Review `docs/rust-native-agents.md` and `docs/agentica-improvements.md`
* Optional: Agentica Python SDK for comparison

**Lesson**

* **How each session contributes to the agent framework:**

  | Session | Contribution to Agents |
  |---------|----------------------|
  | 1-2 (Shapes) | Tool input/output type checking |
  | 3 (Functors) | Sandboxing as functor, proxy transformations |
  | 3.5 (Coproducts) | Scope merging for multi-source capabilities |
  | 3.6 (Capabilities) | Trait-based `HasDatabase`, `HasLLM` bounds |
  | 4-5 (Diagrams) | Agent execution traces as diagrams |
  | 6 (Tracing) | Zero-cost debug tracing for agent calls |
  | 7-10 (Autodiff) | Credit assignment through agent traces |
  | 11-14 (Probability) | Uncertainty in tool outputs, Bayesian reasoning |
  | 15-16 (DisCoCat) | Compositional prompt understanding |
  | 17-18 (Games) | Multi-agent incentive alignment |
  | 18.5 (Lifetimes) | Parallel-safe agent execution |
  | 19 (Operads) | Multi-tool orchestration with arity constraints |

* **The Rust advantage over Python Agentica:**
  * Compile-time capability checking (not runtime `KeyError`)
  * Zero-cost isolation via borrow checker (not WASM sandbox)
  * Proven parallel safety via `Send + Sync` (not hope-based `asyncio`)
  * Type-level composition `Then<A, B>` (not runtime dict merge)

**Build (commit 21) — `agents` crate**

* Add `crates/agents/src/lib.rs` with these modules:
  * `scope.rs` — Capability traits and scope composition
  * `tool.rs` — Tool trait and implementations
  * `llm.rs` — LLM client abstraction
  * `agent.rs` — Core agent loop
  * `trace.rs` — Execution tracing as diagrams
  * `orchestrator.rs` — Multi-agent coordination

### Part 1: Core Agent Types

```rust
// crates/agents/src/lib.rs

pub mod scope;
pub mod tool;
pub mod llm;
pub mod agent;
pub mod trace;
pub mod orchestrator;

pub use scope::{HasDatabase, HasCache, HasLLM, CombinedScope};
pub use tool::{Tool, ToolRegistry, ToolCall, ToolResult};
pub use llm::{LLMClient, Message, CompletionRequest};
pub use agent::{Agent, AgentConfig, AgentError};
pub use trace::{AgentTrace, TraceNode, TracedAgent};
pub use orchestrator::{Orchestrator, OrchestratorConfig};
```

### Part 2: Yoneda-Style Capability Scope (from Session 3.6)

```rust
// crates/agents/src/scope.rs
//
// Yoneda-style capabilities: extensible without modifying core code.
// A capability is defined by what requests it handles, not by a name.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

/// Error when a capability can't handle a request
#[derive(Debug, Clone)]
pub enum CapabilityError {
    NotFound { request_type: &'static str },
    HandlerFailed { message: String },
}

// ============================================================
// Core Traits: Request/Response Pattern (Yoneda-style)
// ============================================================

/// A request defines an operation and its response type.
/// This is the "morphism" in the Yoneda sense — capabilities are
/// characterized by which requests they can handle.
pub trait Request: Send + 'static {
    type Response: Send + 'static;
    fn name() -> &'static str;
}

/// Marker trait for capabilities
pub trait Capability: Send + Sync + 'static {
    fn capability_name(&self) -> &'static str;
}

/// A capability that handles requests of type R.
/// "Handles<R>" = "has a morphism from R into this capability"
pub trait Handles<R: Request>: Capability {
    fn handle(&self, req: R) -> Result<R::Response, CapabilityError>;
}

// ============================================================
// Built-in Request Types (users can add more without modifying core!)
// ============================================================

/// Database query
pub struct DbQuery(pub String);
impl Request for DbQuery {
    type Response = Vec<String>;
    fn name() -> &'static str { "DbQuery" }
}

/// Cache get
pub struct CacheGet(pub String);
impl Request for CacheGet {
    type Response = Option<String>;
    fn name() -> &'static str { "CacheGet" }
}

/// Cache set
pub struct CacheSet { pub key: String, pub value: String, pub ttl_secs: u64 }
impl Request for CacheSet {
    type Response = ();
    fn name() -> &'static str { "CacheSet" }
}

/// LLM completion
pub struct LlmComplete { pub prompt: String, pub max_tokens: usize }
impl Request for LlmComplete {
    type Response = String;
    fn name() -> &'static str { "LlmComplete" }
}

/// Tool invocation (for agent tool calls)
pub struct ToolInvoke { pub tool_name: String, pub args: serde_json::Value }
impl Request for ToolInvoke {
    type Response = serde_json::Value;
    fn name() -> &'static str { "ToolInvoke" }
}

// ============================================================
// Type-erased handler infrastructure
// ============================================================

trait AnyHandler: Send + Sync {
    fn handle_any(&self, req: Box<dyn Any + Send>) -> Result<Box<dyn Any + Send>, CapabilityError>;
}

struct HandlerWrapper<C, R> {
    capability: Arc<C>,
    _phantom: PhantomData<R>,
}

impl<C, R> AnyHandler for HandlerWrapper<C, R>
where
    C: Handles<R> + 'static,
    R: Request,
{
    fn handle_any(&self, req: Box<dyn Any + Send>) -> Result<Box<dyn Any + Send>, CapabilityError> {
        let req = req.downcast::<R>()
            .map_err(|_| CapabilityError::HandlerFailed {
                message: "Request type mismatch".into()
            })?;
        let response = self.capability.handle(*req)?;
        Ok(Box::new(response))
    }
}

// ============================================================
// CapabilityScope: The Yoneda-style registry
// ============================================================

/// Dynamic capability registry using Yoneda-style discovery.
///
/// Instead of hardcoded `HasDatabase`, `HasCache` traits, capabilities
/// are registered by the Request types they handle. This allows:
/// - Adding new capabilities without modifying core
/// - Runtime capability discovery
/// - Type-safe responses (Response type is known at compile time)
#[derive(Default)]
pub struct CapabilityScope {
    handlers: HashMap<TypeId, Box<dyn AnyHandler>>,
}

impl CapabilityScope {
    pub fn new() -> Self {
        Self { handlers: HashMap::new() }
    }

    /// Register a capability for a specific request type
    pub fn register<C, R>(&mut self, capability: Arc<C>)
    where
        C: Handles<R> + 'static,
        R: Request,
    {
        let wrapper = HandlerWrapper {
            capability,
            _phantom: PhantomData::<R>,
        };
        self.handlers.insert(TypeId::of::<R>(), Box::new(wrapper));
    }

    /// Yoneda-style discovery: can this scope handle request R?
    pub fn can_handle<R: Request>(&self) -> bool {
        self.handlers.contains_key(&TypeId::of::<R>())
    }

    /// Dispatch a request — finds the handler and invokes it
    pub fn dispatch<R: Request>(&self, req: R) -> Result<R::Response, CapabilityError> {
        let handler = self.handlers
            .get(&TypeId::of::<R>())
            .ok_or(CapabilityError::NotFound { request_type: R::name() })?;

        let response = handler.handle_any(Box::new(req))?;

        response.downcast::<R::Response>()
            .map(|b| *b)
            .map_err(|_| CapabilityError::HandlerFailed {
                message: "Response type mismatch".into()
            })
    }

    /// Merge scopes (coproduct-style)
    pub fn merge(mut self, other: Self) -> Self {
        for (k, v) in other.handlers {
            self.handlers.insert(k, v);
        }
        self
    }
}

// ============================================================
// Example: A database capability
// ============================================================

pub struct PostgresDb {
    connection_string: String,
}

impl Capability for PostgresDb {
    fn capability_name(&self) -> &'static str { "PostgresDb" }
}

impl Handles<DbQuery> for PostgresDb {
    fn handle(&self, req: DbQuery) -> Result<Vec<String>, CapabilityError> {
        // Real implementation would query the database
        Ok(vec![format!("Result for: {}", req.0)])
    }
}

// ============================================================
// Example: An LLM capability
// ============================================================

pub struct ClaudeClient {
    api_key: String,
}

impl Capability for ClaudeClient {
    fn capability_name(&self) -> &'static str { "ClaudeClient" }
}

impl Handles<LlmComplete> for ClaudeClient {
    fn handle(&self, req: LlmComplete) -> Result<String, CapabilityError> {
        // Real implementation would call Claude API
        Ok(format!("Response to: {} (max {})", req.prompt, req.max_tokens))
    }
}

// ============================================================
// Usage example
// ============================================================

fn build_scope() -> CapabilityScope {
    let mut scope = CapabilityScope::new();

    let db = Arc::new(PostgresDb { connection_string: "...".into() });
    let llm = Arc::new(ClaudeClient { api_key: "...".into() });

    // Register capabilities by the requests they handle
    scope.register::<_, DbQuery>(db);
    scope.register::<_, LlmComplete>(llm);

    scope
}

fn use_scope(scope: &CapabilityScope) -> Result<(), CapabilityError> {
    // Yoneda-style: we don't ask "do you have a database?"
    // We ask "can you handle DbQuery?"
    if scope.can_handle::<DbQuery>() {
        let results = scope.dispatch(DbQuery("SELECT * FROM users".into()))?;
        println!("Got {} results", results.len());
    }

    // Type-safe: dispatch returns the correct Response type
    let completion: String = scope.dispatch(LlmComplete {
        prompt: "Hello".into(),
        max_tokens: 100,
    })?;

    Ok(())
}
```

### Part 3: Tool System (from Sessions 4-5, 19)

```rust
// crates/agents/src/tool.rs

use compositional_core::{Shape, Diagram, CoreError};
use std::collections::HashMap;

/// A tool that an agent can call
pub trait Tool: Send + Sync {
    /// Tool name for LLM function calling
    fn name(&self) -> &str;

    /// JSON schema for the tool's parameters
    fn schema(&self) -> serde_json::Value;

    /// Input/output shapes for diagram composition
    fn input_shape(&self) -> Shape;
    fn output_shape(&self) -> Shape;

    /// Execute the tool
    fn call(&self, args: serde_json::Value) -> Result<ToolResult, ToolError>;
}

/// Result of a tool call
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub output: serde_json::Value,
    pub shape: Shape,
}

/// Tool registry with compile-time capability requirements
pub struct ToolRegistry<S> {
    tools: HashMap<String, Box<dyn Tool>>,
    _scope: std::marker::PhantomData<S>,
}

impl<S> ToolRegistry<S> {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            _scope: std::marker::PhantomData,
        }
    }

    /// Register a tool that requires specific capabilities
    pub fn register<T: Tool + 'static>(&mut self, tool: T) {
        self.tools.insert(tool.name().to_string(), Box::new(tool));
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    pub fn list(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Convert tool calls to a Diagram for tracing (Session 4-5)
    pub fn to_diagram(&self, calls: &[ToolCall]) -> Diagram<ToolOp> {
        // Build diagram from tool call sequence
        todo!()
    }
}

/// A tool call request from the LLM
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Diagram operation for tool calls
#[derive(Debug, Clone)]
pub struct ToolOp {
    pub tool_name: String,
    pub input_shape: Shape,
    pub output_shape: Shape,
}
```

### Part 4: LLM Client Abstraction

```rust
// crates/agents/src/llm.rs

use async_trait::async_trait;

/// Message in conversation
#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// Request to the LLM
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub messages: Vec<Message>,
    pub tools: Option<Vec<serde_json::Value>>,
    pub max_tokens: usize,
}

/// Response from the LLM
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    pub message: Message,
    pub usage: Usage,
}

#[derive(Debug, Clone)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

/// LLM client trait — implement for Claude, GPT, etc.
#[async_trait]
pub trait LLMClient: Send + Sync {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LLMError>;
}

/// Mock LLM for testing
pub struct MockLLM {
    responses: Vec<CompletionResponse>,
}

impl MockLLM {
    pub fn with_responses(responses: Vec<CompletionResponse>) -> Self {
        Self { responses }
    }
}

#[async_trait]
impl LLMClient for MockLLM {
    async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, LLMError> {
        // Return next response in sequence
        todo!()
    }
}
```

### Part 5: Core Agent Loop (from Sessions 3.6, 18.5)

```rust
// crates/agents/src/agent.rs

use crate::{scope::*, tool::*, llm::*, trace::*};
use std::sync::Arc;

/// Agent configuration
#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub system_prompt: String,
    pub max_turns: usize,
    pub max_tool_calls: usize,
}

/// Core agent — generic over scope capabilities
pub struct Agent<'scope, S> {
    scope: &'scope S,
    config: AgentConfig,
    tools: ToolRegistry<S>,
}

impl<'scope, S> Agent<'scope, S>
where
    S: HasLLM + Send + Sync,
{
    pub fn new(scope: &'scope S, config: AgentConfig) -> Self {
        Self {
            scope,
            config,
            tools: ToolRegistry::new(),
        }
    }

    pub fn with_tool<T: Tool + 'static>(mut self, tool: T) -> Self {
        self.tools.register(tool);
        self
    }

    /// Run the agent loop
    pub async fn run(&self, task: &str) -> Result<AgentResult, AgentError> {
        let mut messages = vec![
            Message {
                role: Role::System,
                content: self.config.system_prompt.clone(),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: Role::User,
                content: task.to_string(),
                tool_calls: None,
                tool_call_id: None,
            },
        ];

        let mut trace = AgentTrace::new();
        let mut turns = 0;

        loop {
            if turns >= self.config.max_turns {
                return Err(AgentError::MaxTurnsExceeded);
            }

            // Call LLM
            let request = CompletionRequest {
                messages: messages.clone(),
                tools: Some(self.tool_schemas()),
                max_tokens: 4096,
            };

            let response = self.scope.llm().complete(request).await?;
            trace.record_llm_call(&response);

            // Check for tool calls
            if let Some(tool_calls) = &response.message.tool_calls {
                for call in tool_calls {
                    let result = self.execute_tool(call)?;
                    trace.record_tool_call(call, &result);

                    messages.push(Message {
                        role: Role::Tool,
                        content: serde_json::to_string(&result.output)?,
                        tool_calls: None,
                        tool_call_id: Some(call.name.clone()),
                    });
                }
                messages.push(response.message);
            } else {
                // No tool calls — agent is done
                return Ok(AgentResult {
                    response: response.message.content,
                    trace,
                });
            }

            turns += 1;
        }
    }

    fn tool_schemas(&self) -> Vec<serde_json::Value> {
        self.tools.list()
            .iter()
            .filter_map(|name| self.tools.get(name))
            .map(|t| t.schema())
            .collect()
    }

    fn execute_tool(&self, call: &ToolCall) -> Result<ToolResult, AgentError> {
        let tool = self.tools.get(&call.name)
            .ok_or_else(|| AgentError::UnknownTool(call.name.clone()))?;
        tool.call(call.arguments.clone())
            .map_err(AgentError::ToolError)
    }
}

/// Result of agent execution
#[derive(Debug)]
pub struct AgentResult {
    pub response: String,
    pub trace: AgentTrace,
}
```

### Part 6: Execution Tracing as Diagrams (from Sessions 4-6, 9)

```rust
// crates/agents/src/trace.rs

use compositional_core::{Diagram, Shape};
use crate::{tool::*, llm::*};
use std::time::{Duration, Instant};

/// Complete trace of agent execution
#[derive(Debug)]
pub struct AgentTrace {
    pub nodes: Vec<TraceNode>,
    pub start_time: Instant,
    pub total_duration: Duration,
}

/// A single node in the trace
#[derive(Debug)]
pub enum TraceNode {
    LLMCall {
        prompt_tokens: usize,
        completion_tokens: usize,
        duration: Duration,
    },
    ToolCall {
        tool_name: String,
        input: serde_json::Value,
        output: serde_json::Value,
        duration: Duration,
    },
}

impl AgentTrace {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            start_time: Instant::now(),
            total_duration: Duration::ZERO,
        }
    }

    pub fn record_llm_call(&mut self, response: &CompletionResponse) {
        self.nodes.push(TraceNode::LLMCall {
            prompt_tokens: response.usage.prompt_tokens,
            completion_tokens: response.usage.completion_tokens,
            duration: Duration::ZERO, // Would measure actual duration
        });
    }

    pub fn record_tool_call(&mut self, call: &ToolCall, result: &ToolResult) {
        self.nodes.push(TraceNode::ToolCall {
            tool_name: call.name.clone(),
            input: call.arguments.clone(),
            output: result.output.clone(),
            duration: Duration::ZERO,
        });
    }

    /// Convert trace to a Diagram for visualization and credit assignment
    /// This connects to Session 9's backprop-style credit assignment
    pub fn to_diagram(&self) -> Diagram<TraceOp> {
        // Build diagram from trace nodes
        // Each node becomes a box, edges show data flow
        todo!()
    }

    /// Credit assignment: which tool calls contributed to success/failure?
    /// Uses reverse-mode traversal like Session 9's autodiff
    pub fn credit_assignment(&self, outcome_score: f32) -> Vec<(String, f32)> {
        // Propagate credit backward through trace
        // Similar to VJP but for discrete tool calls
        todo!()
    }

    /// Render trace as ASCII (from Session 6)
    pub fn render_ascii(&self) -> String {
        let mut output = String::new();
        for (i, node) in self.nodes.iter().enumerate() {
            match node {
                TraceNode::LLMCall { prompt_tokens, completion_tokens, .. } => {
                    output.push_str(&format!(
                        "[{}] LLM: {} prompt + {} completion tokens\n",
                        i, prompt_tokens, completion_tokens
                    ));
                }
                TraceNode::ToolCall { tool_name, .. } => {
                    output.push_str(&format!("[{}] Tool: {}\n", i, tool_name));
                }
            }
        }
        output
    }
}

/// Operation type for trace diagrams
#[derive(Debug, Clone)]
pub enum TraceOp {
    LLM,
    Tool(String),
}
```

### Part 7: Multi-Agent Orchestration (from Sessions 17-19)

```rust
// crates/agents/src/orchestrator.rs

use crate::{agent::*, scope::*};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Orchestrator configuration
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    pub max_agents: usize,
    pub timeout_secs: u64,
}

/// Multi-agent orchestrator with operadic composition (Session 19)
pub struct Orchestrator<S> {
    scope: Arc<S>,
    config: OrchestratorConfig,
}

impl<S> Orchestrator<S>
where
    S: HasLLM + Send + Sync + 'static,
{
    pub fn new(scope: Arc<S>, config: OrchestratorConfig) -> Self {
        Self { scope, config }
    }

    /// Run agents in parallel with proven safety (Session 18.5)
    /// Compiler enforces S: Send + Sync
    pub async fn parallel<A, B>(
        &self,
        agent_a: impl FnOnce(Arc<S>) -> A + Send + 'static,
        agent_b: impl FnOnce(Arc<S>) -> B + Send + 'static,
    ) -> (A, B)
    where
        A: Send + 'static,
        B: Send + 'static,
    {
        let scope_a = self.scope.clone();
        let scope_b = self.scope.clone();

        let (a, b) = tokio::join!(
            tokio::spawn(async move { agent_a(scope_a) }),
            tokio::spawn(async move { agent_b(scope_b) }),
        );

        (a.unwrap(), b.unwrap())
    }

    /// Sequential composition with type checking (Session 5's `then`)
    pub async fn sequence<A, B, R1, R2>(
        &self,
        first: impl FnOnce(Arc<S>) -> R1 + Send + 'static,
        second: impl FnOnce(Arc<S>, R1) -> R2 + Send + 'static,
    ) -> R2
    where
        R1: Send + 'static,
        R2: Send + 'static,
    {
        let scope = self.scope.clone();
        let result1 = first(scope.clone());
        second(scope, result1)
    }

    /// Operadic composition: run exactly N agents and combine results
    /// Arity is enforced at compile time via tuple types (Session 19)
    pub async fn combine_3<A1, A2, A3, R>(
        &self,
        agent1: impl FnOnce(Arc<S>) -> A1 + Send + 'static,
        agent2: impl FnOnce(Arc<S>) -> A2 + Send + 'static,
        agent3: impl FnOnce(Arc<S>) -> A3 + Send + 'static,
        combiner: impl FnOnce(A1, A2, A3) -> R,
    ) -> R
    where
        A1: Send + 'static,
        A2: Send + 'static,
        A3: Send + 'static,
    {
        let (r1, r2, r3) = tokio::join!(
            tokio::spawn({ let s = self.scope.clone(); async move { agent1(s) } }),
            tokio::spawn({ let s = self.scope.clone(); async move { agent2(s) } }),
            tokio::spawn({ let s = self.scope.clone(); async move { agent3(s) } }),
        );

        combiner(r1.unwrap(), r2.unwrap(), r3.unwrap())
    }
}
```

### Part 8: Example Tools

```rust
// crates/agents/src/tools/search.rs

use crate::{tool::*, scope::*};

/// Web search tool
pub struct SearchTool;

impl Tool for SearchTool {
    fn name(&self) -> &str { "search" }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        })
    }

    fn input_shape(&self) -> Shape {
        Shape::scalar(TypeId("string"))
    }

    fn output_shape(&self) -> Shape {
        Shape::vector(TypeId("string"), 10)  // Up to 10 results
    }

    fn call(&self, args: serde_json::Value) -> Result<ToolResult, ToolError> {
        let query = args["query"].as_str()
            .ok_or(ToolError::InvalidArgs("missing query".into()))?;

        // Mock search results
        Ok(ToolResult {
            output: serde_json::json!([
                {"title": "Result 1", "snippet": "..."},
                {"title": "Result 2", "snippet": "..."},
            ]),
            shape: self.output_shape(),
        })
    }
}

/// Database query tool — requires HasDatabase capability
pub struct DbQueryTool<S> {
    scope: std::sync::Arc<S>,
}

impl<S: HasDatabase> DbQueryTool<S> {
    pub fn new(scope: std::sync::Arc<S>) -> Self {
        Self { scope }
    }
}

impl<S: HasDatabase + Send + Sync + 'static> Tool for DbQueryTool<S> {
    fn name(&self) -> &str { "db_query" }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "db_query",
                "description": "Query the database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "SQL query to execute"
                        }
                    },
                    "required": ["sql"]
                }
            }
        })
    }

    fn input_shape(&self) -> Shape {
        Shape::scalar(TypeId("string"))
    }

    fn output_shape(&self) -> Shape {
        Shape::vector(TypeId("row"), 100)  // Up to 100 rows
    }

    fn call(&self, args: serde_json::Value) -> Result<ToolResult, ToolError> {
        let sql = args["sql"].as_str()
            .ok_or(ToolError::InvalidArgs("missing sql".into()))?;

        // This compiles because S: HasDatabase
        let rows = self.scope.db().query(sql)?;

        Ok(ToolResult {
            output: serde_json::json!(rows),
            shape: self.output_shape(),
        })
    }
}
```

### Part 9: Demo — Research Agent

```rust
// demos/src/agents.rs

use agents::{*, scope::*, tool::*, llm::*};
use std::sync::Arc;

/// Demo: Research agent that searches and summarizes
pub async fn demo_research_agent() {
    // Create scope with mock implementations
    let scope = Arc::new(ProductionScope::new(
        MockDatabase::new(),
        MockCache::new(),
        MockLLM::with_responses(vec![
            // First response: call search tool
            CompletionResponse {
                message: Message {
                    role: Role::Assistant,
                    content: String::new(),
                    tool_calls: Some(vec![ToolCall {
                        name: "search".into(),
                        arguments: serde_json::json!({"query": "Rust async"}),
                    }]),
                    tool_call_id: None,
                },
                usage: Usage { prompt_tokens: 100, completion_tokens: 50 },
            },
            // Second response: summarize results
            CompletionResponse {
                message: Message {
                    role: Role::Assistant,
                    content: "Rust's async/await provides zero-cost abstractions...".into(),
                    tool_calls: None,
                    tool_call_id: None,
                },
                usage: Usage { prompt_tokens: 200, completion_tokens: 100 },
            },
        ]),
    ));

    // Create agent with tools
    let agent = Agent::new(
        &*scope,
        AgentConfig {
            system_prompt: "You are a helpful research assistant.".into(),
            max_turns: 10,
            max_tool_calls: 5,
        },
    )
    .with_tool(SearchTool);

    // Run agent
    let result = agent.run("What is Rust's async model?").await.unwrap();

    println!("Response: {}", result.response);
    println!("\nTrace:\n{}", result.trace.render_ascii());
}

/// Demo: Parallel agents with orchestrator
pub async fn demo_parallel_agents() {
    let scope = Arc::new(ProductionScope::new(/* ... */));

    let orchestrator = Orchestrator::new(
        scope,
        OrchestratorConfig {
            max_agents: 3,
            timeout_secs: 30,
        },
    );

    // Run research + analysis in parallel (Session 18.5 safety)
    let (research_result, analysis_result) = orchestrator.parallel(
        |s| async move {
            let agent = Agent::new(&*s, research_config());
            agent.run("Research topic X").await
        },
        |s| async move {
            let agent = Agent::new(&*s, analysis_config());
            agent.run("Analyze data Y").await
        },
    ).await;

    println!("Research: {:?}", research_result);
    println!("Analysis: {:?}", analysis_result);
}
```

### Tests

```rust
// crates/agents/tests/integration.rs

use agents::*;

#[tokio::test]
async fn test_agent_tool_call() {
    let scope = TestScope::new();
    let agent = Agent::new(&scope, test_config())
        .with_tool(SearchTool);

    let result = agent.run("Search for X").await.unwrap();
    assert!(!result.response.is_empty());
    assert!(result.trace.nodes.len() >= 2);  // At least LLM + tool
}

#[tokio::test]
async fn test_capability_bounds() {
    // This test verifies compile-time capability checking
    struct MinimalScope;
    impl HasLLM for MinimalScope {
        fn llm(&self) -> &dyn LLMClient { todo!() }
    }

    // Agent only needs HasLLM, so this compiles
    let _agent = Agent::new(&MinimalScope, test_config());

    // DbQueryTool needs HasDatabase, so this would NOT compile:
    // let _tool = DbQueryTool::new(Arc::new(MinimalScope));
    // error: MinimalScope doesn't implement HasDatabase
}

#[tokio::test]
async fn test_parallel_safety() {
    let scope = Arc::new(TestScope::new());
    let orchestrator = Orchestrator::new(scope, test_orchestrator_config());

    // Compiler proves this is safe via Send + Sync bounds
    let (a, b) = orchestrator.parallel(
        |s| { /* uses &S */ 1 },
        |s| { /* uses &S */ 2 },
    ).await;

    assert_eq!(a + b, 3);
}

#[tokio::test]
async fn test_trace_to_diagram() {
    let scope = TestScope::new();
    let agent = Agent::new(&scope, test_config())
        .with_tool(SearchTool)
        .with_tool(CalculatorTool);

    let result = agent.run("Search and calculate").await.unwrap();

    // Trace converts to diagram (Session 4-5)
    let diagram = result.trace.to_diagram();
    assert!(diagram.validate().is_ok());
}

#[test]
fn test_scope_composition() {
    struct DbScope;
    impl HasDatabase for DbScope {
        fn db(&self) -> &dyn Database { todo!() }
    }

    struct LLMScope;
    impl HasLLM for LLMScope {
        fn llm(&self) -> &dyn LLMClient { todo!() }
    }

    // Combine scopes — result has both capabilities
    let combined = CombinedScope { a: DbScope, b: LLMScope };

    // This compiles because combined: HasDatabase + HasLLM
    fn needs_both<S: HasDatabase + HasLLM>(_s: &S) {}
    needs_both(&combined);
}
```

### CLI Integration

```rust
// demos/src/main.rs (add to existing)

#[derive(Subcommand)]
enum Commands {
    // ... existing commands ...

    /// Run agent demos
    Agent {
        #[command(subcommand)]
        demo: AgentDemo,
    },
}

#[derive(Subcommand)]
enum AgentDemo {
    /// Single research agent
    Research,
    /// Parallel agents with orchestrator
    Parallel,
    /// Show trace as diagram
    Trace,
}

// In main():
Commands::Agent { demo } => match demo {
    AgentDemo::Research => agents::demo_research_agent().await,
    AgentDemo::Parallel => agents::demo_parallel_agents().await,
    AgentDemo::Trace => agents::demo_trace_diagram().await,
},
```

### Done-when

* [ ] `cargo run -p demos -- agent research` runs successfully
* [ ] `cargo run -p demos -- agent parallel` demonstrates parallel safety
* [ ] `cargo run -p demos -- agent trace` shows execution trace as diagram
* [ ] All capability trait bounds compile correctly
* [ ] Missing capabilities cause compile errors (not runtime errors)
* [ ] Parallel execution uses `Send + Sync` bounds
* [ ] Trace converts to valid `Diagram<TraceOp>`
* [ ] Credit assignment propagates through trace

---

## What You've Built

Congratulations! You've built a **Rust-native agent framework** that demonstrates:

1. **Compile-time capability checking** — Missing tools/resources are compile errors
2. **Zero-cost isolation** — Borrow checker, not sandboxes
3. **Proven parallel safety** — `Send + Sync`, not hope
4. **Type-level composition** — Diagrams and operads ensure correct wiring
5. **Credit assignment** — Backprop-style analysis of agent traces

This isn't "Agentica ported to Rust" — it's a fundamentally different architecture where **the type system IS the categorical structure**.

The abstractions from Sessions 1-20 directly enable this:
- Shapes → Tool type checking
- Diagrams → Execution traces
- Functors → Sandboxing/proxying
- Operads → Multi-tool orchestration
- `Send + Sync` → Parallel safety

**Next steps:**
- Integrate with real LLM APIs (Claude, GPT)
- Add persistent memory via the probability track (Sessions 11-14)
- Implement learned tool selection using autodiff (Sessions 7-10)
- Build multi-agent games with incentive alignment (Sessions 17-18)

---

If you want, I can also paste a **starter `demos/src/main.rs` with Clap subcommands** and the **exact trait bounds** for `Diagram<O>` so the generic composition + petgraph cloning stays painless in Rust.