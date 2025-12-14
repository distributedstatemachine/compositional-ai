//! # NLP - Compositional Natural Language Processing (Sessions 15-16)
//!
//! This crate implements categorical approaches to natural language processing,
//! including pregroup grammars and DisCoCat (Distributional Compositional Categorical) models.
//!
//! ## Core Concepts
//!
//! - **Pregroup grammars**: Types reduce to determine grammaticality
//! - **Type reductions**: N · Nʳ → 1 and Nˡ · N → 1 (cups in string diagrams)
//! - **Grammaticality**: A sentence is grammatical iff types reduce to S
//! - **DisCoCat**: Tensor networks from grammatical structure
//!
//! ## Example: Parsing "Alice loves Bob"
//!
//! ```rust
//! use compositional_nlp::pregroup::Grammar;
//!
//! let grammar = Grammar::english_basic();
//!
//! // Parse and check grammaticality
//! let result = grammar.parse(&["Alice", "loves", "Bob"]).unwrap();
//! assert!(result.is_grammatical());
//!
//! // See the type reductions
//! println!("{}", result.trace());
//! ```

pub mod discocat;
mod error;
pub mod pregroup;
pub mod semantics;

pub use discocat::{
    build_verb_tensor, cosine_similarity, spearman_correlation, Baselines, DisCoCat,
    EvaluationResult, SentenceSemantics, SentenceTensor, Tensor3, VerbConstructionMethod,
};
pub use error::NlpError;
pub use pregroup::{
    extract_cups, AtomicType, BasicType, Cup, Grammar, ParseResult, PregroupType, ReductionStep,
    TypedWord,
};
pub use semantics::{
    bilinear_form, inner_product, matrix_vector_mult, vector_matrix_mult, Semantics, WordTensor,
};
