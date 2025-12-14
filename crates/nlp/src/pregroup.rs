//! Pregroup Grammars and Type Reductions
//!
//! Pregroups provide a type-theoretic approach to syntax where grammaticality
//! is determined by type reduction. This is the foundation of DisCoCat models.
//!
//! # Key Concepts
//!
//! - **Basic types**: N (noun), S (sentence)
//! - **Adjoints**: Nˡ (left adjoint), Nʳ (right adjoint)
//! - **Reduction rules**: N · Nʳ → 1 and Nˡ · N → 1
//! - **Grammaticality**: A sentence is grammatical iff types reduce to S
//!
//! # Example
//!
//! ```rust
//! use compositional_nlp::pregroup::{PregroupType, TypedWord, Grammar};
//!
//! let mut grammar = Grammar::new();
//!
//! // Define a simple lexicon
//! grammar.add_word("Alice", PregroupType::noun());
//! grammar.add_word("runs", PregroupType::intransitive_verb());
//!
//! // Parse "Alice runs"
//! let sentence = vec!["Alice", "runs"];
//! let result = grammar.parse(&sentence).unwrap();
//! assert!(result.is_grammatical());
//! ```

use crate::NlpError;
use std::collections::HashMap;
use std::fmt;

/// Basic type in a pregroup grammar.
///
/// We support the standard linguistic types:
/// - N: noun
/// - S: sentence
/// - Custom types for extensibility
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BasicType {
    /// Noun type
    N,
    /// Sentence type
    S,
    /// Custom type (for extensibility)
    Custom(String),
}

impl fmt::Display for BasicType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BasicType::N => write!(f, "N"),
            BasicType::S => write!(f, "S"),
            BasicType::Custom(s) => write!(f, "{}", s),
        }
    }
}

/// An atomic type in a pregroup (basic type with adjoint level).
///
/// The adjoint level indicates:
/// - 0: the basic type itself (e.g., N)
/// - positive n: right adjoint applied n times (e.g., Nʳ for n=1)
/// - negative n: left adjoint applied |n| times (e.g., Nˡ for n=-1)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AtomicType {
    /// The basic type
    pub base: BasicType,
    /// Adjoint level: 0 = base, >0 = right adjoints, <0 = left adjoints
    pub adjoint: i32,
}

impl AtomicType {
    /// Create a new atomic type with given adjoint level.
    pub fn new(base: BasicType, adjoint: i32) -> Self {
        Self { base, adjoint }
    }

    /// Create a basic type (adjoint level 0).
    pub fn basic(base: BasicType) -> Self {
        Self { base, adjoint: 0 }
    }

    /// Get the right adjoint (Xʳ).
    pub fn right_adjoint(&self) -> Self {
        Self {
            base: self.base.clone(),
            adjoint: self.adjoint + 1,
        }
    }

    /// Get the left adjoint (Xˡ).
    pub fn left_adjoint(&self) -> Self {
        Self {
            base: self.base.clone(),
            adjoint: self.adjoint - 1,
        }
    }

    /// Check if this type can reduce with another on the right.
    ///
    /// X · Xʳ → 1 (right reduction)
    pub fn can_reduce_right(&self, other: &AtomicType) -> bool {
        self.base == other.base && self.adjoint + 1 == other.adjoint
    }

    /// Check if this type can reduce with another on the left.
    ///
    /// Xˡ · X → 1 (left reduction)
    /// Self is Xˡ (adjoint -1), other is X (adjoint 0)
    /// So self.adjoint + 1 == other.adjoint
    pub fn can_reduce_left(&self, other: &AtomicType) -> bool {
        self.base == other.base && self.adjoint + 1 == other.adjoint
    }
}

impl fmt::Display for AtomicType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.base)?;
        match self.adjoint.cmp(&0) {
            std::cmp::Ordering::Greater => {
                for _ in 0..self.adjoint {
                    write!(f, "ʳ")?;
                }
            }
            std::cmp::Ordering::Less => {
                for _ in 0..(-self.adjoint) {
                    write!(f, "ˡ")?;
                }
            }
            std::cmp::Ordering::Equal => {}
        }
        Ok(())
    }
}

/// A pregroup type (tensor product of atomic types).
///
/// Represents types like:
/// - N (single noun)
/// - Nʳ · S · Nˡ (transitive verb)
/// - N · Nˡ (adjective)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PregroupType {
    /// The atomic types in this tensor product (left to right)
    pub factors: Vec<AtomicType>,
}

impl PregroupType {
    /// Create a new pregroup type from atomic factors.
    pub fn new(factors: Vec<AtomicType>) -> Self {
        Self { factors }
    }

    /// Create the unit type (empty tensor product).
    pub fn unit() -> Self {
        Self { factors: vec![] }
    }

    /// Create a single atomic type.
    pub fn atomic(base: BasicType, adjoint: i32) -> Self {
        Self {
            factors: vec![AtomicType::new(base, adjoint)],
        }
    }

    /// Create the noun type N.
    pub fn noun() -> Self {
        Self::atomic(BasicType::N, 0)
    }

    /// Create the sentence type S.
    pub fn sentence() -> Self {
        Self::atomic(BasicType::S, 0)
    }

    /// Create an intransitive verb type: Nʳ · S
    ///
    /// Takes a subject (noun on left), produces a sentence.
    pub fn intransitive_verb() -> Self {
        Self::new(vec![
            AtomicType::new(BasicType::N, 1), // Nʳ
            AtomicType::new(BasicType::S, 0), // S
        ])
    }

    /// Create a transitive verb type: Nʳ · S · Nˡ
    ///
    /// Takes subject (left) and object (right), produces sentence.
    pub fn transitive_verb() -> Self {
        Self::new(vec![
            AtomicType::new(BasicType::N, 1),  // Nʳ
            AtomicType::new(BasicType::S, 0),  // S
            AtomicType::new(BasicType::N, -1), // Nˡ
        ])
    }

    /// Create an adjective type: N · Nˡ
    ///
    /// Modifies a noun on the right.
    pub fn adjective() -> Self {
        Self::new(vec![
            AtomicType::new(BasicType::N, 0),  // N
            AtomicType::new(BasicType::N, -1), // Nˡ
        ])
    }

    /// Create a determiner type: N · Nˡ
    ///
    /// Same as adjective - turns noun into noun phrase.
    pub fn determiner() -> Self {
        Self::adjective()
    }

    /// Create an adverb type: (Nʳ · S) · (Nʳ · S)ˡ = Nʳ · S · Sˡ · N
    ///
    /// Modifies an intransitive verb. Simplified to: S · Sˡ
    pub fn adverb_intransitive() -> Self {
        Self::new(vec![
            AtomicType::new(BasicType::S, 0),  // S
            AtomicType::new(BasicType::S, -1), // Sˡ
        ])
    }

    /// Create a sentence complement verb type: Nʳ · S · Sˡ
    ///
    /// Takes subject (left) and sentence complement (right), produces sentence.
    /// E.g., "thinks" in "Alice thinks Bob runs"
    pub fn sentence_complement_verb() -> Self {
        Self::new(vec![
            AtomicType::new(BasicType::N, 1),  // Nʳ
            AtomicType::new(BasicType::S, 0),  // S
            AtomicType::new(BasicType::S, -1), // Sˡ
        ])
    }

    /// Tensor product of two pregroup types.
    pub fn tensor(&self, other: &PregroupType) -> PregroupType {
        let mut factors = self.factors.clone();
        factors.extend(other.factors.clone());
        PregroupType { factors }
    }

    /// Check if this is the unit type.
    pub fn is_unit(&self) -> bool {
        self.factors.is_empty()
    }

    /// Check if this is exactly the sentence type S.
    pub fn is_sentence(&self) -> bool {
        self.factors.len() == 1
            && self.factors[0].base == BasicType::S
            && self.factors[0].adjoint == 0
    }

    /// Check if this is exactly the noun type N.
    pub fn is_noun(&self) -> bool {
        self.factors.len() == 1
            && self.factors[0].base == BasicType::N
            && self.factors[0].adjoint == 0
    }

    /// Get the number of atomic factors.
    pub fn len(&self) -> usize {
        self.factors.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.factors.is_empty()
    }
}

impl fmt::Display for PregroupType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.factors.is_empty() {
            write!(f, "1")
        } else {
            let parts: Vec<String> = self.factors.iter().map(|t| t.to_string()).collect();
            write!(f, "{}", parts.join(" · "))
        }
    }
}

/// A word with its assigned pregroup type.
#[derive(Debug, Clone)]
pub struct TypedWord {
    /// The word string
    pub word: String,
    /// The pregroup type
    pub typ: PregroupType,
}

impl TypedWord {
    /// Create a new typed word.
    pub fn new(word: &str, typ: PregroupType) -> Self {
        Self {
            word: word.to_string(),
            typ,
        }
    }
}

impl fmt::Display for TypedWord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} : {}", self.word, self.typ)
    }
}

/// A single reduction step.
#[derive(Debug, Clone)]
pub struct ReductionStep {
    /// Position where reduction occurred
    pub position: usize,
    /// The two types that reduced
    pub left: AtomicType,
    pub right: AtomicType,
    /// Type before reduction
    pub before: PregroupType,
    /// Type after reduction
    pub after: PregroupType,
}

impl fmt::Display for ReductionStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "At position {}: {} · {} → 1\n  {} → {}",
            self.position, self.left, self.right, self.before, self.after
        )
    }
}

/// Result of parsing a sentence.
#[derive(Debug, Clone)]
pub struct ParseResult {
    /// The original sentence
    pub sentence: Vec<String>,
    /// The typed words
    pub typed_words: Vec<TypedWord>,
    /// Initial concatenated type
    pub initial_type: PregroupType,
    /// Reduction steps taken
    pub reductions: Vec<ReductionStep>,
    /// Final type after all reductions
    pub final_type: PregroupType,
}

impl ParseResult {
    /// Check if the parse is grammatical (reduces to S).
    pub fn is_grammatical(&self) -> bool {
        self.final_type.is_sentence()
    }

    /// Check if the parse produces a noun phrase.
    pub fn is_noun_phrase(&self) -> bool {
        self.final_type.is_noun()
    }

    /// Display a detailed trace of the parse.
    pub fn trace(&self) -> String {
        let mut result = String::new();

        result.push_str("=== Parse Trace ===\n\n");

        result.push_str("Sentence: ");
        result.push_str(&self.sentence.join(" "));
        result.push_str("\n\n");

        result.push_str("Word types:\n");
        for tw in &self.typed_words {
            result.push_str(&format!("  {}\n", tw));
        }
        result.push('\n');

        result.push_str(&format!("Initial type: {}\n\n", self.initial_type));

        if self.reductions.is_empty() {
            result.push_str("No reductions possible.\n");
        } else {
            result.push_str("Reductions:\n");
            for (i, step) in self.reductions.iter().enumerate() {
                result.push_str(&format!("  Step {}: {}\n", i + 1, step));
            }
        }

        result.push_str(&format!("\nFinal type: {}\n", self.final_type));

        if self.is_grammatical() {
            result.push_str("Result: GRAMMATICAL (reduces to S)\n");
        } else if self.is_noun_phrase() {
            result.push_str("Result: NOUN PHRASE (reduces to N)\n");
        } else {
            result.push_str("Result: UNGRAMMATICAL\n");
        }

        result
    }
}

/// A pregroup grammar with a lexicon.
#[derive(Debug, Clone)]
pub struct Grammar {
    /// Lexicon: word → type
    lexicon: HashMap<String, PregroupType>,
}

impl Grammar {
    /// Create a new empty grammar.
    pub fn new() -> Self {
        Self {
            lexicon: HashMap::new(),
        }
    }

    /// Create a grammar with a basic English lexicon.
    pub fn english_basic() -> Self {
        let mut grammar = Self::new();

        // Nouns (proper nouns)
        for name in ["Alice", "Bob", "Charlie", "Eve"] {
            grammar.add_word(name, PregroupType::noun());
        }

        // Common nouns
        for noun in ["dog", "cat", "man", "woman", "book", "table", "mouse"] {
            grammar.add_word(noun, PregroupType::noun());
        }

        // Determiners
        for det in ["the", "a", "an", "every", "some"] {
            grammar.add_word(det, PregroupType::determiner());
        }

        // Adjectives
        for adj in ["big", "small", "red", "blue", "happy", "sad"] {
            grammar.add_word(adj, PregroupType::adjective());
        }

        // Intransitive verbs
        for verb in ["runs", "sleeps", "walks", "jumps", "falls"] {
            grammar.add_word(verb, PregroupType::intransitive_verb());
        }

        // Transitive verbs
        for verb in [
            "loves", "hates", "sees", "knows", "chases", "adores", "bites",
        ] {
            grammar.add_word(verb, PregroupType::transitive_verb());
        }

        // Sentence complement verbs
        for verb in ["thinks", "believes", "knows_that", "says"] {
            grammar.add_word(verb, PregroupType::sentence_complement_verb());
        }

        grammar
    }

    /// Add a word to the lexicon.
    pub fn add_word(&mut self, word: &str, typ: PregroupType) {
        self.lexicon.insert(word.to_string(), typ);
    }

    /// Get the type of a word (if in lexicon).
    pub fn get_type(&self, word: &str) -> Option<&PregroupType> {
        self.lexicon.get(word)
    }

    /// Type a sentence (assign types to each word).
    pub fn type_sentence(&self, words: &[&str]) -> Result<Vec<TypedWord>, NlpError> {
        let mut typed = Vec::new();
        for word in words {
            let typ = self
                .lexicon
                .get(*word)
                .ok_or_else(|| NlpError::UnknownWord {
                    word: word.to_string(),
                })?;
            typed.push(TypedWord::new(word, typ.clone()));
        }
        Ok(typed)
    }

    /// Concatenate types of typed words.
    pub fn concatenate_types(&self, typed_words: &[TypedWord]) -> PregroupType {
        let mut result = PregroupType::unit();
        for tw in typed_words {
            result = result.tensor(&tw.typ);
        }
        result
    }

    /// Perform one reduction step (if possible).
    ///
    /// Returns the position and resulting type, or None if no reduction possible.
    fn reduce_once(
        &self,
        typ: &PregroupType,
    ) -> Option<(usize, AtomicType, AtomicType, PregroupType)> {
        for i in 0..typ.factors.len().saturating_sub(1) {
            let left = &typ.factors[i];
            let right = &typ.factors[i + 1];

            // Check X · Xʳ → 1 (right reduction)
            if left.can_reduce_right(right) {
                let mut new_factors = typ.factors.clone();
                new_factors.remove(i + 1);
                new_factors.remove(i);
                return Some((
                    i,
                    left.clone(),
                    right.clone(),
                    PregroupType::new(new_factors),
                ));
            }

            // Check Xˡ · X → 1 (left reduction)
            if left.can_reduce_left(right) {
                let mut new_factors = typ.factors.clone();
                new_factors.remove(i + 1);
                new_factors.remove(i);
                return Some((
                    i,
                    left.clone(),
                    right.clone(),
                    PregroupType::new(new_factors),
                ));
            }
        }
        None
    }

    /// Reduce a type as far as possible.
    pub fn reduce(&self, typ: &PregroupType) -> (PregroupType, Vec<ReductionStep>) {
        let mut current = typ.clone();
        let mut steps = Vec::new();

        while let Some((pos, left, right, new_type)) = self.reduce_once(&current) {
            steps.push(ReductionStep {
                position: pos,
                left,
                right,
                before: current.clone(),
                after: new_type.clone(),
            });
            current = new_type;
        }

        (current, steps)
    }

    /// Parse a sentence and return the result.
    pub fn parse(&self, words: &[&str]) -> Result<ParseResult, NlpError> {
        let typed_words = self.type_sentence(words)?;
        let initial_type = self.concatenate_types(&typed_words);
        let (final_type, reductions) = self.reduce(&initial_type);

        Ok(ParseResult {
            sentence: words.iter().map(|s| s.to_string()).collect(),
            typed_words,
            initial_type,
            reductions,
            final_type,
        })
    }

    /// Check if a sentence is grammatical.
    pub fn is_grammatical(&self, words: &[&str]) -> Result<bool, NlpError> {
        let result = self.parse(words)?;
        Ok(result.is_grammatical())
    }
}

impl Default for Grammar {
    fn default() -> Self {
        Self::new()
    }
}

/// Represents a reduction as a "cup" in a string diagram.
///
/// Cups connect adjacent wires that can be contracted.
#[derive(Debug, Clone)]
pub struct Cup {
    /// Position of the left wire
    pub left_pos: usize,
    /// Position of the right wire
    pub right_pos: usize,
    /// The base type being contracted
    pub base_type: BasicType,
}

/// Extract cups (contractions) from a parse result.
///
/// This is useful for building tensor network diagrams.
pub fn extract_cups(result: &ParseResult) -> Vec<Cup> {
    let mut cups = Vec::new();

    // Track the original positions before reductions
    let mut position_map: Vec<usize> = (0..result.initial_type.factors.len()).collect();

    for step in &result.reductions {
        let left_original = position_map[step.position];
        let right_original = position_map[step.position + 1];

        cups.push(Cup {
            left_pos: left_original,
            right_pos: right_original,
            base_type: step.left.base.clone(),
        });

        // Update position map after removal
        position_map.remove(step.position + 1);
        position_map.remove(step.position);
    }

    cups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_type_display() {
        let n = AtomicType::basic(BasicType::N);
        assert_eq!(n.to_string(), "N");

        let nr = n.right_adjoint();
        assert_eq!(nr.to_string(), "Nʳ");

        let nl = n.left_adjoint();
        assert_eq!(nl.to_string(), "Nˡ");

        let nrr = nr.right_adjoint();
        assert_eq!(nrr.to_string(), "Nʳʳ");
    }

    #[test]
    fn test_pregroup_type_display() {
        let tv = PregroupType::transitive_verb();
        assert_eq!(tv.to_string(), "Nʳ · S · Nˡ");

        let unit = PregroupType::unit();
        assert_eq!(unit.to_string(), "1");

        let noun = PregroupType::noun();
        assert_eq!(noun.to_string(), "N");
    }

    #[test]
    fn test_reduction_right() {
        let n = AtomicType::basic(BasicType::N);
        let nr = AtomicType::new(BasicType::N, 1);

        assert!(n.can_reduce_right(&nr));
        assert!(!nr.can_reduce_right(&n));
    }

    #[test]
    fn test_reduction_left() {
        let n = AtomicType::basic(BasicType::N);
        let nl = AtomicType::new(BasicType::N, -1);

        assert!(nl.can_reduce_left(&n));
        assert!(!n.can_reduce_left(&nl));
    }

    #[test]
    fn test_intransitive_sentence() {
        let grammar = Grammar::english_basic();

        // "Alice runs" should be grammatical
        let result = grammar.parse(&["Alice", "runs"]).unwrap();
        assert!(result.is_grammatical());
    }

    #[test]
    fn test_transitive_sentence() {
        let grammar = Grammar::english_basic();

        // "Alice loves Bob" should be grammatical
        let result = grammar.parse(&["Alice", "loves", "Bob"]).unwrap();
        assert!(result.is_grammatical());
    }

    #[test]
    fn test_determiner_noun() {
        let grammar = Grammar::english_basic();

        // "the dog" should produce a noun phrase
        let result = grammar.parse(&["the", "dog"]).unwrap();
        assert!(result.is_noun_phrase());
    }

    #[test]
    fn test_adjective_noun() {
        let grammar = Grammar::english_basic();

        // "big dog" should produce a noun phrase
        let result = grammar.parse(&["big", "dog"]).unwrap();
        assert!(result.is_noun_phrase());
    }

    #[test]
    fn test_complex_sentence() {
        let grammar = Grammar::english_basic();

        // "the big dog runs" should be grammatical
        let result = grammar.parse(&["the", "big", "dog", "runs"]).unwrap();
        assert!(result.is_grammatical());
    }

    #[test]
    fn test_subject_object_sentence() {
        let grammar = Grammar::english_basic();

        // "Alice sees the big cat" should be grammatical
        let result = grammar
            .parse(&["Alice", "sees", "the", "big", "cat"])
            .unwrap();
        assert!(result.is_grammatical());
    }

    #[test]
    fn test_ungrammatical_order() {
        let grammar = Grammar::english_basic();

        // "loves Alice Bob" should not reduce to S
        let result = grammar.parse(&["loves", "Alice", "Bob"]).unwrap();
        assert!(!result.is_grammatical());
    }

    #[test]
    fn test_sentence_complement() {
        let grammar = Grammar::english_basic();

        // "Alice thinks Bob runs" should be grammatical
        let result = grammar.parse(&["Alice", "thinks", "Bob", "runs"]).unwrap();
        assert!(result.is_grammatical());
    }

    #[test]
    fn test_unknown_word() {
        let grammar = Grammar::english_basic();

        let result = grammar.parse(&["Alice", "fnords", "Bob"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_cups() {
        let grammar = Grammar::english_basic();
        let result = grammar.parse(&["Alice", "loves", "Bob"]).unwrap();
        let cups = extract_cups(&result);

        // Should have 2 cups: N·Nʳ and Nˡ·N
        assert_eq!(cups.len(), 2);
    }
}
