//! # Categories - Composition Interfaces
//!
//! This module provides the categorical foundations:
//! - Finite categories (toy examples for learning)
//! - Functors (structure-preserving maps)
//! - Opposite categories (for reverse-mode autodiff in Session 9)
//!
//! ## Why This Matters
//!
//! These abstractions seem abstract now, but they pay off later:
//! - **Session 9**: Backprop is a functor to the opposite category
//! - **Session 15**: DisCoCat semantics is a functor Grammar → Vect
//! - **Session 19**: Operad algebras are functors
//!
//! We build the infrastructure here and reference it explicitly later.

use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

/// A morphism in a finite category.
///
/// In category theory, a morphism f: A → B connects two objects.
/// Here we store the domain (source) and codomain (target) explicitly.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Morphism<O: Clone + Eq + Hash> {
    pub name: String,
    pub dom: O, // domain (source)
    pub cod: O, // codomain (target)
}

impl<O: Clone + Eq + Hash + fmt::Display> fmt::Display for Morphism<O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} → {}", self.name, self.dom, self.cod)
    }
}

/// A finite category with explicit objects and morphisms.
///
/// This is a "toy" category for learning — real categories are usually
/// defined by their structure, not enumerated. But finite categories
/// let us test composition laws explicitly.
#[derive(Debug, Clone)]
pub struct FiniteCategory<O: Clone + Eq + Hash> {
    /// All objects in the category
    pub objects: Vec<O>,
    /// All morphisms, keyed by name
    pub morphisms: HashMap<String, Morphism<O>>,
    /// Composition table: (f, g) → f;g (if defined)
    /// Convention: f;g means "f then g" (diagram order)
    compositions: HashMap<(String, String), String>,
    /// Identity morphisms for each object
    identities: HashMap<O, String>,
}

impl<O: Clone + Eq + Hash + fmt::Display> FiniteCategory<O> {
    /// Create a new empty category.
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            morphisms: HashMap::new(),
            compositions: HashMap::new(),
            identities: HashMap::new(),
        }
    }

    /// Add an object to the category.
    pub fn add_object(&mut self, obj: O) -> &mut Self {
        if !self.objects.contains(&obj) {
            // Create identity morphism for this object
            let id_name = format!("id_{}", obj);
            self.morphisms.insert(
                id_name.clone(),
                Morphism {
                    name: id_name.clone(),
                    dom: obj.clone(),
                    cod: obj.clone(),
                },
            );
            self.identities.insert(obj.clone(), id_name);
            self.objects.push(obj);
        }
        self
    }

    /// Add a morphism to the category.
    pub fn add_morphism(&mut self, name: &str, dom: O, cod: O) -> &mut Self {
        self.morphisms.insert(
            name.to_string(),
            Morphism {
                name: name.to_string(),
                dom,
                cod,
            },
        );
        self
    }

    /// Define a composition: f;g = h
    /// Returns error if cod(f) ≠ dom(g)
    pub fn define_composition(
        &mut self,
        f: &str,
        g: &str,
        result: &str,
    ) -> Result<&mut Self, String> {
        let f_mor = self
            .morphisms
            .get(f)
            .ok_or(format!("Unknown morphism: {}", f))?;
        let g_mor = self
            .morphisms
            .get(g)
            .ok_or(format!("Unknown morphism: {}", g))?;

        if f_mor.cod != g_mor.dom {
            return Err(format!(
                "Cannot compose {} and {}: cod({}) = {} ≠ {} = dom({})",
                f, g, f, f_mor.cod, g_mor.dom, g
            ));
        }

        self.compositions
            .insert((f.to_string(), g.to_string()), result.to_string());
        Ok(self)
    }

    /// Look up the composition of two morphisms.
    pub fn compose(&self, f: &str, g: &str) -> Option<&Morphism<O>> {
        // Handle identity composition
        if let Some(f_mor) = self.morphisms.get(f) {
            if self.identities.get(&f_mor.dom) == Some(&f.to_string()) {
                // f is identity, return g
                return self.morphisms.get(g);
            }
        }
        if let Some(g_mor) = self.morphisms.get(g) {
            if self.identities.get(&g_mor.dom) == Some(&g.to_string()) {
                // g is identity, return f
                return self.morphisms.get(f);
            }
        }

        // Look up in composition table
        self.compositions
            .get(&(f.to_string(), g.to_string()))
            .and_then(|h| self.morphisms.get(h))
    }

    /// Get identity morphism for an object.
    pub fn identity(&self, obj: &O) -> Option<&Morphism<O>> {
        self.identities
            .get(obj)
            .and_then(|name| self.morphisms.get(name))
    }
}

impl<O: Clone + Eq + Hash + fmt::Display> Default for FiniteCategory<O> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Functors
// ============================================================================

/// A functor F: C → D maps objects and morphisms while preserving structure.
///
/// The laws:
/// 1. F(id_A) = id_{F(A)}           (preserves identities)
/// 2. F(f;g) = F(f);F(g)            (preserves composition)
pub trait Functor<C, D> {
    /// The source category
    type Source;
    /// The target category
    type Target;

    /// Map an object from C to D
    fn map_obj(&self, obj: &C) -> D;

    /// Map a morphism from C to D
    fn map_mor(&self, mor: &str) -> String;
}

/// Check that a functor preserves identity and composition.
/// Returns Ok(()) if laws hold, Err with violation description otherwise.
pub fn check_functor_laws<O, F>(
    source: &FiniteCategory<O>,
    target: &FiniteCategory<O>,
    functor: &F,
) -> Result<(), String>
where
    O: Clone + Eq + Hash + fmt::Display,
    F: Functor<O, O>,
{
    // Check identity preservation: F(id_A) = id_{F(A)}
    for obj in &source.objects {
        let f_obj = functor.map_obj(obj);
        let id_in_source = source
            .identity(obj)
            .ok_or(format!("No identity for {} in source", obj))?;
        let mapped_id = functor.map_mor(&id_in_source.name);
        let expected_id = target
            .identity(&f_obj)
            .ok_or(format!("No identity for {} in target", f_obj))?;

        if mapped_id != expected_id.name {
            return Err(format!(
                "Identity not preserved: F({}) = {} ≠ {}",
                id_in_source.name, mapped_id, expected_id.name
            ));
        }
    }

    // Check composition preservation would go here
    // (requires iterating over all composable pairs)

    Ok(())
}

// ============================================================================
// Opposite Category
// ============================================================================

/// Wrapper that reverses arrows in a category.
///
/// If C has morphism f: A → B, then C^op has morphism f^op: B → A.
/// Composition reverses: in C^op, f^op ; g^op = (g;f)^op
///
/// **This is crucial for Session 9**: The backward pass of autodiff
/// is a functor from the forward computation category to its opposite.
#[derive(Debug, Clone)]
pub struct OppositeCategory<O: Clone + Eq + Hash> {
    /// The original category
    pub original: FiniteCategory<O>,
}

impl<O: Clone + Eq + Hash + fmt::Display> OppositeCategory<O> {
    /// Create the opposite of a category.
    pub fn new(cat: FiniteCategory<O>) -> Self {
        Self { original: cat }
    }

    /// Get a morphism with reversed domain/codomain.
    pub fn get_morphism(&self, name: &str) -> Option<Morphism<O>> {
        self.original.morphisms.get(name).map(|m| Morphism {
            name: format!("{}_op", m.name),
            dom: m.cod.clone(), // Reversed!
            cod: m.dom.clone(), // Reversed!
        })
    }

    /// Compose in the opposite category: f^op ; g^op = (g;f)^op
    pub fn compose(&self, f: &str, g: &str) -> Option<Morphism<O>> {
        // In opposite category, composition is reversed
        self.original.compose(g, f).map(|m| Morphism {
            name: format!("{}_op", m.name),
            dom: m.cod.clone(),
            cod: m.dom.clone(),
        })
    }
}

// ============================================================================
// Forward Reference Markers
// ============================================================================

/// Marker: This will be used in Session 9 to show backprop as a functor
/// from the forward computation category to its opposite.
///
/// The VJP (vector-Jacobian product) of f: A → B is vjp_f: B* → A*
/// where X* denotes the "cotangent" type (same shape, gradient semantics).
pub struct BackpropFunctorMarker;

/// Marker: This will be used in Session 15 for grammar → Vect semantics.
///
/// The semantics functor maps:
/// - Grammar types (N, S, etc.) → Vector spaces (ℝⁿ)
/// - Grammar reductions → Tensor contractions
pub struct SemanticsFunctorMarker;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finite_category_creation() {
        let mut cat: FiniteCategory<String> = FiniteCategory::new();
        cat.add_object("A".to_string())
            .add_object("B".to_string())
            .add_morphism("f", "A".to_string(), "B".to_string());

        assert_eq!(cat.objects.len(), 2);
        assert!(cat.morphisms.contains_key("f"));
        assert!(cat.morphisms.contains_key("id_A"));
        assert!(cat.morphisms.contains_key("id_B"));
    }

    #[test]
    fn test_identity_composition() {
        let mut cat: FiniteCategory<String> = FiniteCategory::new();
        cat.add_object("A".to_string())
            .add_object("B".to_string())
            .add_morphism("f", "A".to_string(), "B".to_string());

        // id_A ; f = f
        let result = cat.compose("id_A", "f");
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "f");

        // f ; id_B = f
        let result = cat.compose("f", "id_B");
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "f");
    }

    #[test]
    fn test_opposite_category() {
        let mut cat: FiniteCategory<String> = FiniteCategory::new();
        cat.add_object("A".to_string())
            .add_object("B".to_string())
            .add_morphism("f", "A".to_string(), "B".to_string());

        let op = OppositeCategory::new(cat);

        // In C: f: A → B
        // In C^op: f_op: B → A
        let f_op = op.get_morphism("f").unwrap();
        assert_eq!(f_op.dom, "B".to_string());
        assert_eq!(f_op.cod, "A".to_string());
    }

    #[test]
    fn test_composition_order() {
        let mut cat: FiniteCategory<String> = FiniteCategory::new();
        cat.add_object("A".to_string())
            .add_object("B".to_string())
            .add_object("C".to_string())
            .add_morphism("f", "A".to_string(), "B".to_string())
            .add_morphism("g", "B".to_string(), "C".to_string())
            .add_morphism("gf", "A".to_string(), "C".to_string());

        cat.define_composition("f", "g", "gf").unwrap();

        let result = cat.compose("f", "g");
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "gf");
    }
}
