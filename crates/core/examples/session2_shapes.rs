//! Session 2: Shapes - Type-Level Tensor Dimensions
//!
//! Run with: cargo run --example session2_shapes
//!
//! This example demonstrates:
//! - Creating shapes for scalars, vectors, and matrices
//! - Shape compatibility checking
//! - How shapes prevent invalid compositions

use compositional_core::{CoreError, Shape};

fn main() {
    println!("=== Session 2: Shapes ===\n");

    // -------------------------------------------------------------------------
    // Creating Shapes
    // -------------------------------------------------------------------------
    println!("1. Creating Shapes");
    println!("-------------------");

    let scalar = Shape::f32_scalar();
    let vector_128 = Shape::f32_vector(128);
    let vector_256 = Shape::f32_vector(256);
    let matrix_64x128 = Shape::f32_matrix(64, 128);

    println!("Scalar:      {} (rank {})", scalar, scalar.rank());
    println!("Vector(128): {} (rank {})", vector_128, vector_128.rank());
    println!("Vector(256): {} (rank {})", vector_256, vector_256.rank());
    println!(
        "Matrix:      {} (rank {})",
        matrix_64x128,
        matrix_64x128.rank()
    );
    println!();

    // -------------------------------------------------------------------------
    // Shape Compatibility
    // -------------------------------------------------------------------------
    println!("2. Shape Compatibility");
    println!("----------------------");

    // Same shapes are compatible
    let v1 = Shape::f32_vector(128);
    let v2 = Shape::f32_vector(128);
    println!(
        "f32[128] compatible with f32[128]? {}",
        v1.is_compatible(&v2)
    );

    // Different dimensions are incompatible
    let v3 = Shape::f32_vector(256);
    println!(
        "f32[128] compatible with f32[256]? {}",
        v1.is_compatible(&v3)
    );

    // Different ranks are incompatible
    println!(
        "f32[128] compatible with f32[]?    {}",
        v1.is_compatible(&scalar)
    );
    println!();

    // -------------------------------------------------------------------------
    // Why Shapes Matter: Composition Safety
    // -------------------------------------------------------------------------
    println!("3. Why Shapes Matter");
    println!("--------------------");

    // Imagine two neural network layers:
    // Layer 1: f32[784] -> f32[128]  (input layer)
    // Layer 2: f32[128] -> f32[10]   (output layer)

    let layer1_input = Shape::f32_vector(784);
    let layer1_output = Shape::f32_vector(128);
    let layer2_input = Shape::f32_vector(128);
    let layer2_output = Shape::f32_vector(10);

    println!("Layer 1: {} -> {}", layer1_input, layer1_output);
    println!("Layer 2: {} -> {}", layer2_input, layer2_output);
    println!(
        "Can compose? {} (output matches input)",
        layer1_output.is_compatible(&layer2_input)
    );
    println!();

    // But what if we try to compose incompatible layers?
    let bad_layer2_input = Shape::f32_vector(256);
    println!(
        "Bad Layer 2 input: {} (expects 256, not 128)",
        bad_layer2_input
    );
    println!(
        "Can compose? {}",
        layer1_output.is_compatible(&bad_layer2_input)
    );
    println!();

    // -------------------------------------------------------------------------
    // ShapeMismatch Errors
    // -------------------------------------------------------------------------
    println!("4. ShapeMismatch Errors");
    println!("-----------------------");

    let error = CoreError::ShapeMismatch {
        expected: Shape::f32_vector(256),
        got: Shape::f32_vector(128),
    };
    println!("Error: {}", error);
    println!();

    // -------------------------------------------------------------------------
    // Generic Shapes
    // -------------------------------------------------------------------------
    println!("5. Generic Shapes");
    println!("-----------------");

    // Create shapes with arbitrary dimensions using TypeId
    use compositional_core::TypeId;

    let tensor_3d = Shape::new(TypeId("f32"), vec![32, 64, 128]);
    println!("3D Tensor: {} (rank {})", tensor_3d, tensor_3d.rank());

    let tensor_4d = Shape::new(TypeId("f32"), vec![8, 16, 32, 64]);
    println!("4D Tensor: {} (rank {})", tensor_4d, tensor_4d.rank());

    // Custom types
    let embedding = Shape::new(TypeId("embedding"), vec![512]);
    println!("Embedding: {} (rank {})", embedding, embedding.rank());

    println!("\n=== Session 2 Complete ===");
}
