//! # Shape Tests (Session 2)
//!
//! Tests for the shape system:
//! - Constructors (scalar, vector, matrix)
//! - Shape mismatch detection
//! - Static shape conversion
//!
//! These tests verify that "types as interfaces" works correctly.

use compositional_core::shape::{Embedding128, Mat3x3, Shape, StaticShape, TypeId};
use compositional_core::CoreError;

// ============================================================================
// Constructor Tests
// ============================================================================

#[test]
fn test_scalar_constructor() {
    let s = Shape::scalar(TypeId("f32"));
    assert_eq!(s.ty, TypeId("f32"));
    assert_eq!(s.dims, Vec::<usize>::new());
    assert_eq!(s.rank(), 0);
    assert_eq!(s.numel(), 1); // scalar has 1 element
}

#[test]
fn test_vector_constructor() {
    let v = Shape::vector(TypeId("f32"), 128);
    assert_eq!(v.ty, TypeId("f32"));
    assert_eq!(v.dims, vec![128]);
    assert_eq!(v.rank(), 1);
    assert_eq!(v.numel(), 128);
}

#[test]
fn test_matrix_constructor() {
    let m = Shape::matrix(TypeId("f32"), 64, 128);
    assert_eq!(m.ty, TypeId("f32"));
    assert_eq!(m.dims, vec![64, 128]);
    assert_eq!(m.rank(), 2);
    assert_eq!(m.numel(), 64 * 128);
}

#[test]
fn test_generic_constructor() {
    let tensor = Shape::new(TypeId("f64"), vec![2, 3, 4, 5]);
    assert_eq!(tensor.rank(), 4);
    assert_eq!(tensor.numel(), 2 * 3 * 4 * 5);
}

#[test]
fn test_convenience_constructors() {
    let s = Shape::f32_scalar();
    let v = Shape::f32_vector(10);
    let m = Shape::f32_matrix(3, 4);

    assert_eq!(s.ty, TypeId("f32"));
    assert_eq!(v.ty, TypeId("f32"));
    assert_eq!(m.ty, TypeId("f32"));

    assert_eq!(s.dims, Vec::<usize>::new());
    assert_eq!(v.dims, vec![10]);
    assert_eq!(m.dims, vec![3, 4]);
}

// ============================================================================
// Display Tests
// ============================================================================

#[test]
fn test_shape_display() {
    assert_eq!(Shape::f32_scalar().to_string(), "f32[]");
    assert_eq!(Shape::f32_vector(10).to_string(), "f32[10]");
    assert_eq!(Shape::f32_matrix(3, 4).to_string(), "f32[3, 4]");

    let tensor = Shape::new(TypeId("i32"), vec![2, 3, 4]);
    assert_eq!(tensor.to_string(), "i32[2, 3, 4]");
}

#[test]
fn test_typeid_display() {
    let ty = TypeId("embedding");
    assert_eq!(ty.to_string(), "embedding");
}

// ============================================================================
// Compatibility / Mismatch Tests
// ============================================================================

#[test]
fn test_compatible_shapes() {
    let a = Shape::f32_vector(128);
    let b = Shape::f32_vector(128);

    assert!(a.is_compatible(&b));
    assert!(b.is_compatible(&a)); // symmetric
}

#[test]
fn test_incompatible_dims() {
    let a = Shape::f32_vector(128);
    let b = Shape::f32_vector(256);

    assert!(!a.is_compatible(&b));
}

#[test]
fn test_incompatible_rank() {
    let scalar = Shape::f32_scalar();
    let vector = Shape::f32_vector(1);

    // Even though both have 1 element, they're different shapes
    assert!(!scalar.is_compatible(&vector));
}

#[test]
fn test_incompatible_type() {
    let f32_vec = Shape::vector(TypeId("f32"), 128);
    let i32_vec = Shape::vector(TypeId("i32"), 128);

    // Same dims, different type
    assert!(!f32_vec.is_compatible(&i32_vec));
}

#[test]
fn test_shape_mismatch_error() {
    let expected = Shape::f32_vector(128);
    let got = Shape::f32_vector(256);

    let error = CoreError::ShapeMismatch {
        expected: expected.clone(),
        got: got.clone(),
    };

    // Error message should contain both shapes
    let msg = error.to_string();
    assert!(msg.contains("128"));
    assert!(msg.contains("256"));
}

// ============================================================================
// Static Shape Tests
// ============================================================================

#[test]
fn test_mat3x3_static_shape() {
    // Compile-time constant
    assert_eq!(Mat3x3::DIMS, &[3, 3]);

    // Runtime conversion
    let shape = Mat3x3::to_shape();
    assert_eq!(shape.dims, vec![3, 3]);
    assert_eq!(shape.ty, TypeId("f32"));
}

#[test]
fn test_embedding128_static_shape() {
    assert_eq!(Embedding128::DIMS, &[128]);

    let shape = Embedding128::to_shape();
    assert_eq!(shape.dims, vec![128]);
    assert_eq!(shape.ty, TypeId("f32"));
}

#[test]
fn test_static_to_runtime_compatibility() {
    // Static shape converts to runtime shape that's compatible with equivalent runtime shape
    let static_shape = Mat3x3::to_shape();
    let runtime_shape = Shape::f32_matrix(3, 3);

    assert!(static_shape.is_compatible(&runtime_shape));
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_empty_dims_is_scalar() {
    let s1 = Shape::new(TypeId("f32"), vec![]);
    let s2 = Shape::f32_scalar();

    assert!(s1.is_compatible(&s2));
    assert_eq!(s1.rank(), 0);
}

#[test]
fn test_single_dim_one_is_not_scalar() {
    let scalar = Shape::f32_scalar();
    let vec_one = Shape::f32_vector(1);

    // [1] is NOT the same as [] even though both have numel=1
    assert!(!scalar.is_compatible(&vec_one));
    assert_eq!(scalar.rank(), 0);
    assert_eq!(vec_one.rank(), 1);
}

#[test]
fn test_zero_dim_vector() {
    let v = Shape::f32_vector(0);
    assert_eq!(v.numel(), 0);
    assert_eq!(v.rank(), 1);
}

#[test]
fn test_shape_equality() {
    let a = Shape::f32_matrix(3, 4);
    let b = Shape::f32_matrix(3, 4);
    let c = Shape::f32_matrix(4, 3);

    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn test_shape_clone() {
    let original = Shape::f32_matrix(3, 4);
    let cloned = original.clone();

    assert_eq!(original, cloned);
}

#[test]
fn test_shape_hash() {
    use std::collections::HashSet;

    let mut set = HashSet::new();
    set.insert(Shape::f32_vector(10));
    set.insert(Shape::f32_vector(10)); // duplicate
    set.insert(Shape::f32_vector(20));

    assert_eq!(set.len(), 2); // only 2 unique shapes
}
