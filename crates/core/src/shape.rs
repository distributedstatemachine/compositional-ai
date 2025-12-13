//! # Shapes - Types as Objects
//!
//! In category theory, objects are the "types" that morphisms connect.
//! In our system, shapes are the types — they describe tensor dimensions
//! and must match for composition to be valid.
//!
//! ## Design Choices
//!
//! We use runtime shape checking (`Vec<usize>`) rather than compile-time
//! (const generics / typenum) for flexibility with dynamic graphs.
//! See `StaticShape` trait for the compile-time alternative.

use std::fmt;

/// A type identifier for distinguishing different kinds of data.
///
/// Examples: "f32", "image", "embedding", "token_ids"
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeId(pub &'static str);

impl fmt::Display for TypeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A shape describes the dimensions of a tensor or data structure.
///
/// This is an "object" in the categorical sense — morphisms (operations)
/// have input and output shapes, and composition is only valid when
/// output shapes match input shapes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    /// The underlying data type
    pub ty: TypeId,
    /// Dimension sizes (empty = scalar, [n] = vector, [m,n] = matrix, etc.)
    pub dims: Vec<usize>,
}

impl Shape {
    /// Create a new shape with given type and dimensions.
    pub fn new(ty: TypeId, dims: Vec<usize>) -> Self {
        Self { ty, dims }
    }

    /// Create a scalar shape (0-dimensional).
    pub fn scalar(ty: TypeId) -> Self {
        Self { ty, dims: vec![] }
    }

    /// Create a vector shape (1-dimensional).
    pub fn vector(ty: TypeId, len: usize) -> Self {
        Self {
            ty,
            dims: vec![len],
        }
    }

    /// Create a matrix shape (2-dimensional).
    pub fn matrix(ty: TypeId, rows: usize, cols: usize) -> Self {
        Self {
            ty,
            dims: vec![rows, cols],
        }
    }

    /// Convenience: f32 scalar
    pub fn f32_scalar() -> Self {
        Self::scalar(TypeId("f32"))
    }

    /// Convenience: f32 vector
    pub fn f32_vector(len: usize) -> Self {
        Self::vector(TypeId("f32"), len)
    }

    /// Convenience: f32 matrix
    pub fn f32_matrix(rows: usize, cols: usize) -> Self {
        Self::matrix(TypeId("f32"), rows, cols)
    }

    /// Number of dimensions (rank).
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    /// Check if this shape is compatible with another for composition.
    pub fn is_compatible(&self, other: &Shape) -> bool {
        self == other
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.dims.is_empty() {
            write!(f, "{}[]", self.ty)
        } else {
            write!(f, "{}[{}]", self.ty,
                self.dims.iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", "))
        }
    }
}

// ============================================================================
// Type-Level Shapes (Optional/Demonstration)
// ============================================================================

/// Marker trait for compile-time shape checking.
///
/// This demonstrates the alternative to runtime checking.
/// With const generics or typenum, shapes can be verified at compile time.
///
/// Trade-offs:
/// - Compile-time: Catches errors early, but inflexible for dynamic graphs
/// - Runtime: Flexible, but errors only caught during construction/execution
pub trait StaticShape {
    /// The dimensions as a compile-time constant
    const DIMS: &'static [usize];

    /// Convert to a runtime Shape
    fn to_shape() -> Shape;
}

/// Example: A 3x3 matrix type known at compile time.
pub struct Mat3x3;

impl StaticShape for Mat3x3 {
    const DIMS: &'static [usize] = &[3, 3];

    fn to_shape() -> Shape {
        Shape::f32_matrix(3, 3)
    }
}

/// Example: A 128-dimensional embedding vector.
pub struct Embedding128;

impl StaticShape for Embedding128 {
    const DIMS: &'static [usize] = &[128];

    fn to_shape() -> Shape {
        Shape::f32_vector(128)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_shape() {
        let s = Shape::f32_scalar();
        assert_eq!(s.rank(), 0);
        assert_eq!(s.numel(), 1);
        assert_eq!(s.to_string(), "f32[]");
    }

    #[test]
    fn test_vector_shape() {
        let v = Shape::f32_vector(10);
        assert_eq!(v.rank(), 1);
        assert_eq!(v.numel(), 10);
        assert_eq!(v.to_string(), "f32[10]");
    }

    #[test]
    fn test_matrix_shape() {
        let m = Shape::f32_matrix(3, 4);
        assert_eq!(m.rank(), 2);
        assert_eq!(m.numel(), 12);
        assert_eq!(m.to_string(), "f32[3, 4]");
    }

    #[test]
    fn test_shape_compatibility() {
        let a = Shape::f32_vector(10);
        let b = Shape::f32_vector(10);
        let c = Shape::f32_vector(20);

        assert!(a.is_compatible(&b));
        assert!(!a.is_compatible(&c));
    }

    #[test]
    fn test_static_shape() {
        let s = Mat3x3::to_shape();
        assert_eq!(s.dims, vec![3, 3]);
        assert_eq!(Mat3x3::DIMS, &[3, 3]);
    }
}
