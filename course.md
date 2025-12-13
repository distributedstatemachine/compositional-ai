Yep — here's the **20-session Rust capstone plan**, where **each session = one commit**, and each session includes:

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
Session 4-5: Diagrams + monoidal composition
     ↓
Session 6: Rendering ──────────────────────────────────┐
     ↓                                                 │
     ├─────────────────┬─────────────────┬────────────┤
     ↓                 ↓                 ↓            │
Sessions 7-10      Sessions 11-14    Sessions 15-16   │
(Autodiff)         (Probability)     (NLP/DisCoCat)   │
     ↓                 ↓                 ↓            │
     └─────────────────┴─────────────────┘            │
                       ↓                              │
                Sessions 17-18 (Games)                │
                       ↓                              │
                Session 19 (Operads) ←────────────────┘
                       ↓
                Session 20 (Integration)
```

**Checkpoints:**
- ✓ **Checkpoint A (Session 6):** Core diagram system complete — you can build, compose, and visualize any diagram
- ✓ **Checkpoint B (Session 10):** Autodiff working — train a model with gradient descent
- ✓ **Checkpoint C (Session 14):** Probability track complete — causal inference operational
- ✓ **Checkpoint D (Session 18):** Games track complete — mechanism design demos working

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

**Build (commit 5) — `then()` + `tensor()`**

* In `core::diagram` implement:

  * `Diagram::then(&self, rhs: &Diagram<O>) -> Result<Diagram<O>, CoreError>`
  * `Diagram::tensor(&self, rhs: &Diagram<O>) -> Diagram<O>`
  * internal `remap_node_ids` helper (copy graphs + reconnect)
* Tests:

  * compose matching boundaries passes
  * mismatched boundary shapes errors
  * tensor doubles boundary count, preserves shapes

---

## Session 6 — String diagram UX: render + DOT export

**Reading**

* *Seven Sketches*: string diagram reasoning (skim)
* (Optional) A quick Graphviz DOT primer

**Lesson**

* Why diagram rendering matters: you debug by *seeing*
* Rewrite mindset: refactors preserve meaning

**Build (commit 6) — ASCII + DOT**

* Add to `core::diagram`:

  * `render_ascii()` (node list + edges with port indices + shapes)
  * `to_dot()` (Graphviz DOT string)
* Add `demos` CLI:

  * `cargo run -p demos -- render --dot` prints DOT for a tiny example
* Tests: snapshot-like string contains expected nodes/edges

---

# Differentiation / Backprop track (Sessions 7–10)

## Session 7 — Computation graphs as diagrams

**Reading**

* "Reverse-mode autodiff" overview (any short note)
* Optional: Fong/Spivak/Tuyéras *Backprop as Functor* (intro/abstract)

**Lesson**

* Evaluation is a monoidal/categorical fold over a DAG
* Shapes guarantee composability before runtime

**Build (commit 7) — `diff` crate scaffolding + Tensor**

* Add `crates/diff/src/lib.rs`, `tensor.rs`, `ops.rs`
* Implement `Tensor`:

  * `Scalar(f32)`, `Vec(Array1<f32>)`, `Mat(Array2<f32>)`
  * `shape()` -> `Shape`
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

## Session 20 — Polish + one coherent CLI + README story

**Reading**

* Re-skim *Seven Sketches* intro: "composition is the interface"

**Lesson**

* Unification: same core wiring abstraction supports learning/probability/semantics/games
* Present it as a systems tool, not "math homework"

**Build (commit 20) — integration**

* `demos` provides subcommands:

  * `render`, `diff`, `prob`, `nlp`, `games`, `operads`
* Add `README.md`:

  * 1 page narrative + how to run each demo
  * include ASCII render snippets
* Add workspace-wide `cargo test` + clippy cleanups (optional)
* Done-when: `cargo run -p demos -- diff` etc all work

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

If you want, I can also paste a **starter `demos/src/main.rs` with Clap subcommands** and the **exact trait bounds** for `Diagram<O>` so the generic composition + petgraph cloning stays painless in Rust.
