//! Smoke tests for the core crate.
//!
//! These tests verify that the basic infrastructure works:
//! - Shapes can be created and compared
//! - Diagrams can be built and validated
//! - Categories can be defined with composition
//!
//! Session 1 completion criteria: `cargo test` passes.

use compositional_core::cat::{FiniteCategory, OppositeCategory};
use compositional_core::diagram::{Diagram, Node, Port};
use compositional_core::{CoreError, Shape};

// ============================================================================
// Shape Tests
// ============================================================================

#[test]
fn smoke_shapes_exist() {
    let scalar = Shape::f32_scalar();
    let vector = Shape::f32_vector(128);
    let matrix = Shape::f32_matrix(64, 128);

    assert_eq!(scalar.rank(), 0);
    assert_eq!(vector.rank(), 1);
    assert_eq!(matrix.rank(), 2);
}

#[test]
fn smoke_shape_display() {
    let s = Shape::f32_matrix(3, 4);
    let display = format!("{}", s);
    assert!(display.contains("3"));
    assert!(display.contains("4"));
}

#[test]
fn smoke_shape_compatibility() {
    let a = Shape::f32_vector(10);
    let b = Shape::f32_vector(10);
    let c = Shape::f32_vector(20);

    assert!(a.is_compatible(&b));
    assert!(!a.is_compatible(&c));
}

// ============================================================================
// Diagram Tests
// ============================================================================

#[derive(Debug, Clone)]
enum SimpleOp {
    Input,
    Add,
    Output,
}

#[test]
fn smoke_diagram_creation() {
    let diagram: Diagram<SimpleOp> = Diagram::new();
    assert_eq!(diagram.node_count(), 0);
}

#[test]
fn smoke_diagram_add_nodes() {
    let mut diagram: Diagram<SimpleOp> = Diagram::new();

    let input_node = Node::new(
        SimpleOp::Input,
        vec![],
        vec![Port::new(Shape::f32_scalar())],
    );

    let add_node = Node::new(
        SimpleOp::Add,
        vec![
            Port::new(Shape::f32_scalar()),
            Port::new(Shape::f32_scalar()),
        ],
        vec![Port::new(Shape::f32_scalar())],
    );

    diagram.add_node(input_node);
    diagram.add_node(add_node);

    assert_eq!(diagram.node_count(), 2);
}

#[test]
fn smoke_diagram_connect() {
    let mut diagram: Diagram<SimpleOp> = Diagram::new();

    let input1 = Node::new(
        SimpleOp::Input,
        vec![],
        vec![Port::new(Shape::f32_scalar())],
    );
    let input2 = Node::new(
        SimpleOp::Input,
        vec![],
        vec![Port::new(Shape::f32_scalar())],
    );
    let add = Node::new(
        SimpleOp::Add,
        vec![
            Port::new(Shape::f32_scalar()),
            Port::new(Shape::f32_scalar()),
        ],
        vec![Port::new(Shape::f32_scalar())],
    );

    let i1 = diagram.add_node(input1);
    let i2 = diagram.add_node(input2);
    let a = diagram.add_node(add);

    // Connect inputs to add node
    diagram
        .connect(i1, 0, a, 0)
        .expect("Connection should succeed");
    diagram
        .connect(i2, 0, a, 1)
        .expect("Connection should succeed");

    assert_eq!(diagram.edge_count(), 2);
}

#[test]
fn smoke_diagram_shape_mismatch() {
    let mut diagram: Diagram<SimpleOp> = Diagram::new();

    let producer = Node::new(
        SimpleOp::Input,
        vec![],
        vec![Port::new(Shape::f32_scalar())], // produces scalar
    );
    let consumer = Node::new(
        SimpleOp::Output,
        vec![Port::new(Shape::f32_vector(10))], // expects vector
        vec![],
    );

    let p = diagram.add_node(producer);
    let c = diagram.add_node(consumer);

    let result = diagram.connect(p, 0, c, 0);
    assert!(matches!(result, Err(CoreError::ShapeMismatch { .. })));
}

// ============================================================================
// Category Tests
// ============================================================================

#[test]
fn smoke_category_creation() {
    let mut cat: FiniteCategory<String> = FiniteCategory::new();

    cat.add_object("A".to_string())
        .add_object("B".to_string())
        .add_object("C".to_string())
        .add_morphism("f", "A".to_string(), "B".to_string())
        .add_morphism("g", "B".to_string(), "C".to_string());

    assert_eq!(cat.objects.len(), 3);
    // 3 identities + 2 morphisms
    assert_eq!(cat.morphisms.len(), 5);
}

#[test]
fn smoke_category_identity_composition() {
    let mut cat: FiniteCategory<String> = FiniteCategory::new();

    cat.add_object("A".to_string())
        .add_object("B".to_string())
        .add_morphism("f", "A".to_string(), "B".to_string());

    // id_A ; f = f
    let result = cat.compose("id_A", "f");
    assert!(result.is_some());
    assert_eq!(result.unwrap().name, "f");
}

#[test]
fn smoke_opposite_category() {
    let mut cat: FiniteCategory<String> = FiniteCategory::new();

    cat.add_object("A".to_string())
        .add_object("B".to_string())
        .add_morphism("f", "A".to_string(), "B".to_string());

    let op = OppositeCategory::new(cat);

    // In C: f: A → B
    // In C^op: f_op: B → A (arrows reversed)
    let f_op = op.get_morphism("f").expect("Should find f");
    assert_eq!(f_op.dom, "B");
    assert_eq!(f_op.cod, "A");
}

// ============================================================================
// Integration: Diagram validates correctly
// ============================================================================

#[test]
fn smoke_diagram_validates() {
    let mut diagram: Diagram<SimpleOp> = Diagram::new();

    let node = Node::new(
        SimpleOp::Add,
        vec![
            Port::new(Shape::f32_scalar()),
            Port::new(Shape::f32_scalar()),
        ],
        vec![Port::new(Shape::f32_scalar())],
    );

    let idx = diagram.add_node(node);
    diagram.set_inputs(vec![(idx, 0), (idx, 1)]);
    diagram.set_outputs(vec![(idx, 0)]);

    assert!(diagram.validate().is_ok());
}

#[test]
fn smoke_complete_pipeline() {
    // Build a tiny pipeline: two inputs → add → output
    let mut diagram: Diagram<SimpleOp> = Diagram::new();

    let input1 = Node::new(
        SimpleOp::Input,
        vec![],
        vec![Port::new(Shape::f32_scalar())],
    );
    let input2 = Node::new(
        SimpleOp::Input,
        vec![],
        vec![Port::new(Shape::f32_scalar())],
    );
    let add = Node::new(
        SimpleOp::Add,
        vec![
            Port::new(Shape::f32_scalar()),
            Port::new(Shape::f32_scalar()),
        ],
        vec![Port::new(Shape::f32_scalar())],
    );
    let output = Node::new(
        SimpleOp::Output,
        vec![Port::new(Shape::f32_scalar())],
        vec![],
    );

    let i1 = diagram.add_node(input1);
    let i2 = diagram.add_node(input2);
    let a = diagram.add_node(add);
    let o = diagram.add_node(output);

    diagram.connect(i1, 0, a, 0).unwrap();
    diagram.connect(i2, 0, a, 1).unwrap();
    diagram.connect(a, 0, o, 0).unwrap();

    diagram.set_outputs(vec![(a, 0)]);

    assert!(diagram.validate().is_ok());
    assert_eq!(diagram.node_count(), 4);
    assert_eq!(diagram.edge_count(), 3);
}

// ============================================================================
// Session 4: Composition Smoke Tests
// ============================================================================

#[test]
fn smoke_sequential_composition() {
    // f: scalar → vector(10)
    let f = Diagram::from_node(Node::new(
        SimpleOp::Add,
        vec![Port::new(Shape::f32_scalar())],
        vec![Port::new(Shape::f32_vector(10))],
    ));

    // g: vector(10) → vector(20)
    let g = Diagram::from_node(Node::new(
        SimpleOp::Add,
        vec![Port::new(Shape::f32_vector(10))],
        vec![Port::new(Shape::f32_vector(20))],
    ));

    // f ; g should compose
    let fg = f.then(g).expect("composition should succeed");
    assert_eq!(fg.node_count(), 2);
    assert_eq!(fg.edge_count(), 1);
}

#[test]
fn smoke_parallel_composition() {
    // f: A → B
    let f = Diagram::from_node(Node::new(
        SimpleOp::Add,
        vec![Port::new(Shape::f32_scalar())],
        vec![Port::new(Shape::f32_vector(10))],
    ));

    // g: C → D
    let g = Diagram::from_node(Node::new(
        SimpleOp::Add,
        vec![Port::new(Shape::f32_vector(20))],
        vec![Port::new(Shape::f32_vector(30))],
    ));

    // f ⊗ g : (A, C) → (B, D)
    let fg = f.tensor(g);
    assert_eq!(fg.node_count(), 2);
    assert_eq!(fg.edge_count(), 0); // no connection between parallel nodes
    assert_eq!(fg.inputs.len(), 2);
    assert_eq!(fg.outputs.len(), 2);
}

#[test]
fn smoke_from_node() {
    let node = Node::new(
        SimpleOp::Add,
        vec![
            Port::new(Shape::f32_scalar()),
            Port::new(Shape::f32_scalar()),
        ],
        vec![Port::new(Shape::f32_scalar())],
    );

    let diagram = Diagram::from_node(node);
    assert_eq!(diagram.node_count(), 1);
    assert_eq!(diagram.inputs.len(), 2);
    assert_eq!(diagram.outputs.len(), 1);
}
