//! # Yoneda-Style Capabilities (Session 3.6)
//!
//! This module implements an extensible capability system inspired by the Yoneda lemma.
//!
//! ## The Problem with Hardcoded Traits
//!
//! Traditional capability systems enumerate traits upfront:
//! ```rust
//! // Must define a new trait for each capability type
//! trait Database {}
//! trait Cache {}
//!
//! trait HasDatabase { fn db(&self) -> &dyn Database; }
//! trait HasCache { fn cache(&self) -> &dyn Cache; }
//! // What about HasVectorDB? HasMetrics? → Endless proliferation
//! ```
//!
//! ## Yoneda's Insight
//!
//! The Yoneda lemma says: an object X is fully characterized by Hom(−, X) —
//! all morphisms into X. Applied to capabilities:
//!
//! **A capability is defined by what requests it can handle, not by a name.**
//!
//! Instead of asking "is this a Database?", we ask "can this handle `DbQuery`?"
//!
//! ## Design
//!
//! - [`Request`]: Defines an operation and its response type (the "morphism")
//! - [`Capability`]: Marker trait for things that handle requests
//! - [`Handles<R>`]: "This capability handles requests of type R"
//! - [`CapabilityScope`]: Registry that dispatches requests to handlers
//!
//! ## Extensibility
//!
//! Users can define new request types without modifying this module:
//! ```rust
//! use compositional_core::capability::Request;
//!
//! // Define a new request type — no changes to core needed!
//! struct VectorSearch { embedding: Vec<f32>, top_k: usize }
//!
//! impl Request for VectorSearch {
//!     type Response = Vec<(String, f32)>;  // (doc_id, similarity)
//!     fn name() -> &'static str { "VectorSearch" }
//! }
//! // No HasVectorDB trait needed!
//! ```

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

// ============================================================================
// Error Types
// ============================================================================

/// Error when a capability can't handle a request.
#[derive(Debug, Clone)]
pub enum CapabilityError {
    /// No handler registered for this request type
    NotFound { request_type: &'static str },
    /// Handler failed to process the request
    HandlerFailed { message: String },
}

impl fmt::Display for CapabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CapabilityError::NotFound { request_type } => {
                write!(f, "No handler for request type: {}", request_type)
            }
            CapabilityError::HandlerFailed { message } => {
                write!(f, "Handler failed: {}", message)
            }
        }
    }
}

impl Error for CapabilityError {}

// ============================================================================
// Core Traits
// ============================================================================

/// A request defines an operation and its response type.
///
/// This is the "morphism" in the Yoneda sense — capabilities are
/// characterized by which request types they can handle.
///
/// # Example
///
/// ```
/// use compositional_core::capability::Request;
///
/// struct Ping;
/// impl Request for Ping {
///     type Response = String;
///     fn name() -> &'static str { "Ping" }
/// }
/// ```
pub trait Request: Send + 'static {
    /// The type returned when this request is handled
    type Response: Send + 'static;

    /// Human-readable name for error messages
    fn name() -> &'static str;
}

/// Marker trait for capabilities (objects that handle requests).
///
/// A capability can implement [`Handles<R>`] for multiple request types.
pub trait Capability: Send + Sync + 'static {
    /// Human-readable name for this capability
    fn capability_name(&self) -> &'static str;
}

/// A capability that can handle requests of type R.
///
/// In Yoneda terms: "`Handles<R>`" = "has a morphism from R into this capability"
///
/// # Example
///
/// ```
/// use compositional_core::capability::{Request, Capability, Handles, CapabilityError};
///
/// struct Ping;
/// impl Request for Ping {
///     type Response = String;
///     fn name() -> &'static str { "Ping" }
/// }
///
/// struct PingService;
/// impl Capability for PingService {
///     fn capability_name(&self) -> &'static str { "PingService" }
/// }
///
/// impl Handles<Ping> for PingService {
///     fn handle(&self, _req: Ping) -> Result<String, CapabilityError> {
///         Ok("pong".to_string())
///     }
/// }
/// ```
pub trait Handles<R: Request>: Capability {
    /// Handle a request and return the response
    fn handle(&self, req: R) -> Result<R::Response, CapabilityError>;
}

// ============================================================================
// Built-in Request Types
// ============================================================================

/// Database query request.
///
/// Users can implement `Handles<SqlQuery>` for their database types.
#[derive(Debug, Clone)]
pub struct SqlQuery(pub String);

impl Request for SqlQuery {
    type Response = Vec<String>; // Simplified: rows as strings
    fn name() -> &'static str {
        "SqlQuery"
    }
}

/// Cache get request.
#[derive(Debug, Clone)]
pub struct CacheGet(pub String);

impl Request for CacheGet {
    type Response = Option<String>;
    fn name() -> &'static str {
        "CacheGet"
    }
}

/// Cache set request.
#[derive(Debug, Clone)]
pub struct CacheSet {
    pub key: String,
    pub value: String,
}

impl Request for CacheSet {
    type Response = ();
    fn name() -> &'static str {
        "CacheSet"
    }
}

/// LLM completion request.
#[derive(Debug, Clone)]
pub struct LlmComplete {
    pub prompt: String,
    pub max_tokens: usize,
}

impl Request for LlmComplete {
    type Response = String;
    fn name() -> &'static str {
        "LlmComplete"
    }
}

// ============================================================================
// Type-Erased Handler Infrastructure
// ============================================================================

/// Type-erased handler trait for dynamic dispatch.
trait AnyHandler: Send + Sync {
    fn handle_any(&self, req: Box<dyn Any + Send>) -> Result<Box<dyn Any + Send>, CapabilityError>;
}

/// Wrapper that implements AnyHandler for any Handles<R>.
struct HandlerWrapper<C, R> {
    capability: Arc<C>,
    _phantom: PhantomData<fn(R) -> R>,
}

// Implement Send + Sync for HandlerWrapper
// Safety: C is already Send + Sync (required by Capability), and PhantomData is always Send + Sync
unsafe impl<C: Send + Sync, R> Send for HandlerWrapper<C, R> {}
unsafe impl<C: Send + Sync, R> Sync for HandlerWrapper<C, R> {}

impl<C, R> AnyHandler for HandlerWrapper<C, R>
where
    C: Handles<R> + 'static,
    R: Request,
{
    fn handle_any(&self, req: Box<dyn Any + Send>) -> Result<Box<dyn Any + Send>, CapabilityError> {
        let req = req
            .downcast::<R>()
            .map_err(|_| CapabilityError::HandlerFailed {
                message: format!("Request type mismatch, expected {}", R::name()),
            })?;
        let response = self.capability.handle(*req)?;
        Ok(Box::new(response))
    }
}

// ============================================================================
// CapabilityScope
// ============================================================================

/// A registry of capabilities indexed by request type.
///
/// This is the Yoneda-style scope: instead of hardcoded `HasDatabase`, `HasCache`
/// traits, capabilities are registered by the request types they handle.
///
/// # Example
///
/// ```
/// use compositional_core::capability::{
///     CapabilityScope, Request, Capability, Handles, CapabilityError
/// };
/// use std::sync::Arc;
///
/// // Define a simple request
/// struct Greet(String);
/// impl Request for Greet {
///     type Response = String;
///     fn name() -> &'static str { "Greet" }
/// }
///
/// // Define a capability that handles it
/// struct Greeter;
/// impl Capability for Greeter {
///     fn capability_name(&self) -> &'static str { "Greeter" }
/// }
/// impl Handles<Greet> for Greeter {
///     fn handle(&self, req: Greet) -> Result<String, CapabilityError> {
///         Ok(format!("Hello, {}!", req.0))
///     }
/// }
///
/// // Register and dispatch
/// let mut scope = CapabilityScope::new();
/// scope.register::<Greeter, Greet>(Arc::new(Greeter));
///
/// assert!(scope.can_handle::<Greet>());
/// let response = scope.dispatch(Greet("World".to_string())).unwrap();
/// assert_eq!(response, "Hello, World!");
/// ```
#[derive(Default)]
pub struct CapabilityScope {
    handlers: HashMap<TypeId, Box<dyn AnyHandler>>,
}

impl CapabilityScope {
    /// Create an empty capability scope.
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }

    /// Register a capability for a specific request type.
    ///
    /// The capability must implement `Handles<R>` for the request type.
    pub fn register<C, R>(&mut self, capability: Arc<C>)
    where
        C: Handles<R> + 'static,
        R: Request,
    {
        let wrapper = HandlerWrapper {
            capability,
            _phantom: PhantomData,
        };
        self.handlers.insert(TypeId::of::<R>(), Box::new(wrapper));
    }

    /// Check if this scope can handle requests of type R.
    ///
    /// This is Yoneda-style discovery: instead of asking "do you have a database?",
    /// we ask "can you handle SqlQuery?"
    pub fn can_handle<R: Request>(&self) -> bool {
        self.handlers.contains_key(&TypeId::of::<R>())
    }

    /// Dispatch a request to its handler.
    ///
    /// Returns `NotFound` if no handler is registered for this request type.
    pub fn dispatch<R: Request>(&self, req: R) -> Result<R::Response, CapabilityError> {
        let handler = self
            .handlers
            .get(&TypeId::of::<R>())
            .ok_or(CapabilityError::NotFound {
                request_type: R::name(),
            })?;

        let response = handler.handle_any(Box::new(req))?;

        response
            .downcast::<R::Response>()
            .map(|b| *b)
            .map_err(|_| CapabilityError::HandlerFailed {
                message: "Response type mismatch".into(),
            })
    }

    /// Merge two scopes (coproduct-style, other wins on conflict).
    pub fn merge(mut self, other: Self) -> Self {
        for (k, v) in other.handlers {
            self.handlers.insert(k, v);
        }
        self
    }

    /// Number of registered handlers.
    pub fn len(&self) -> usize {
        self.handlers.len()
    }

    /// Check if scope has no handlers.
    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }
}

impl fmt::Debug for CapabilityScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CapabilityScope")
            .field("handler_count", &self.handlers.len())
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Test capability: echoes strings
    struct EchoService;

    impl Capability for EchoService {
        fn capability_name(&self) -> &'static str {
            "EchoService"
        }
    }

    // Custom request type for testing
    struct Echo(String);

    impl Request for Echo {
        type Response = String;
        fn name() -> &'static str {
            "Echo"
        }
    }

    impl Handles<Echo> for EchoService {
        fn handle(&self, req: Echo) -> Result<String, CapabilityError> {
            Ok(req.0)
        }
    }

    // Another request type
    struct Reverse(String);

    impl Request for Reverse {
        type Response = String;
        fn name() -> &'static str {
            "Reverse"
        }
    }

    impl Handles<Reverse> for EchoService {
        fn handle(&self, req: Reverse) -> Result<String, CapabilityError> {
            Ok(req.0.chars().rev().collect())
        }
    }

    #[test]
    fn test_dispatch_returns_correct_response() {
        let mut scope = CapabilityScope::new();
        scope.register::<EchoService, Echo>(Arc::new(EchoService));

        let response = scope.dispatch(Echo("hello".to_string())).unwrap();
        assert_eq!(response, "hello");
    }

    #[test]
    fn test_dispatch_not_found_for_unregistered() {
        let scope = CapabilityScope::new();

        let result = scope.dispatch(Echo("hello".to_string()));
        assert!(matches!(result, Err(CapabilityError::NotFound { .. })));
    }

    #[test]
    fn test_can_handle_reflects_registration() {
        let mut scope = CapabilityScope::new();

        assert!(!scope.can_handle::<Echo>());

        scope.register::<EchoService, Echo>(Arc::new(EchoService));

        assert!(scope.can_handle::<Echo>());
        assert!(!scope.can_handle::<Reverse>()); // Not registered yet
    }

    #[test]
    fn test_same_capability_multiple_requests() {
        // One capability can handle multiple request types
        let mut scope = CapabilityScope::new();
        let service = Arc::new(EchoService);

        scope.register::<EchoService, Echo>(service.clone());
        scope.register::<EchoService, Reverse>(service);

        assert!(scope.can_handle::<Echo>());
        assert!(scope.can_handle::<Reverse>());

        let echo_result = scope.dispatch(Echo("hello".to_string())).unwrap();
        assert_eq!(echo_result, "hello");

        let reverse_result = scope.dispatch(Reverse("hello".to_string())).unwrap();
        assert_eq!(reverse_result, "olleh");
    }

    #[test]
    fn test_extensibility_new_request_type() {
        // Users can define new request types without modifying core
        struct CustomRequest {
            value: i32,
        }

        impl Request for CustomRequest {
            type Response = i32;
            fn name() -> &'static str {
                "CustomRequest"
            }
        }

        struct Doubler;

        impl Capability for Doubler {
            fn capability_name(&self) -> &'static str {
                "Doubler"
            }
        }

        impl Handles<CustomRequest> for Doubler {
            fn handle(&self, req: CustomRequest) -> Result<i32, CapabilityError> {
                Ok(req.value * 2)
            }
        }

        let mut scope = CapabilityScope::new();
        scope.register::<Doubler, CustomRequest>(Arc::new(Doubler));

        let result = scope.dispatch(CustomRequest { value: 21 }).unwrap();
        assert_eq!(result, 42);
    }

    #[test]
    fn test_merge_combines_handlers() {
        let mut scope_a = CapabilityScope::new();
        scope_a.register::<EchoService, Echo>(Arc::new(EchoService));

        let mut scope_b = CapabilityScope::new();
        scope_b.register::<EchoService, Reverse>(Arc::new(EchoService));

        let merged = scope_a.merge(scope_b);

        assert!(merged.can_handle::<Echo>());
        assert!(merged.can_handle::<Reverse>());
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_merge_right_wins_on_conflict() {
        struct ServiceA;
        impl Capability for ServiceA {
            fn capability_name(&self) -> &'static str {
                "ServiceA"
            }
        }
        impl Handles<Echo> for ServiceA {
            fn handle(&self, _: Echo) -> Result<String, CapabilityError> {
                Ok("from A".to_string())
            }
        }

        struct ServiceB;
        impl Capability for ServiceB {
            fn capability_name(&self) -> &'static str {
                "ServiceB"
            }
        }
        impl Handles<Echo> for ServiceB {
            fn handle(&self, _: Echo) -> Result<String, CapabilityError> {
                Ok("from B".to_string())
            }
        }

        let mut scope_a = CapabilityScope::new();
        scope_a.register::<ServiceA, Echo>(Arc::new(ServiceA));

        let mut scope_b = CapabilityScope::new();
        scope_b.register::<ServiceB, Echo>(Arc::new(ServiceB));

        let merged = scope_a.merge(scope_b);

        // B should win
        let result = merged.dispatch(Echo("test".to_string())).unwrap();
        assert_eq!(result, "from B");
    }

    #[test]
    fn test_builtin_sql_query() {
        struct MockDb;

        impl Capability for MockDb {
            fn capability_name(&self) -> &'static str {
                "MockDb"
            }
        }

        impl Handles<SqlQuery> for MockDb {
            fn handle(&self, req: SqlQuery) -> Result<Vec<String>, CapabilityError> {
                Ok(vec![format!("Result for: {}", req.0)])
            }
        }

        let mut scope = CapabilityScope::new();
        scope.register::<MockDb, SqlQuery>(Arc::new(MockDb));

        let result = scope
            .dispatch(SqlQuery("SELECT * FROM users".to_string()))
            .unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].contains("SELECT * FROM users"));
    }

    #[test]
    fn test_builtin_cache_operations() {
        use std::sync::Mutex;

        struct MockCache {
            data: Mutex<HashMap<String, String>>,
        }

        impl MockCache {
            fn new() -> Self {
                Self {
                    data: Mutex::new(HashMap::new()),
                }
            }
        }

        impl Capability for MockCache {
            fn capability_name(&self) -> &'static str {
                "MockCache"
            }
        }

        impl Handles<CacheGet> for MockCache {
            fn handle(&self, req: CacheGet) -> Result<Option<String>, CapabilityError> {
                Ok(self.data.lock().unwrap().get(&req.0).cloned())
            }
        }

        impl Handles<CacheSet> for MockCache {
            fn handle(&self, req: CacheSet) -> Result<(), CapabilityError> {
                self.data.lock().unwrap().insert(req.key, req.value);
                Ok(())
            }
        }

        let cache = Arc::new(MockCache::new());
        let mut scope = CapabilityScope::new();
        scope.register::<MockCache, CacheGet>(cache.clone());
        scope.register::<MockCache, CacheSet>(cache);

        // Initially empty
        let result = scope.dispatch(CacheGet("key".to_string())).unwrap();
        assert!(result.is_none());

        // Set a value
        scope
            .dispatch(CacheSet {
                key: "key".to_string(),
                value: "value".to_string(),
            })
            .unwrap();

        // Now it exists
        let result = scope.dispatch(CacheGet("key".to_string())).unwrap();
        assert_eq!(result, Some("value".to_string()));
    }

    #[test]
    fn test_scope_len_and_is_empty() {
        let mut scope = CapabilityScope::new();
        assert!(scope.is_empty());
        assert_eq!(scope.len(), 0);

        scope.register::<EchoService, Echo>(Arc::new(EchoService));
        assert!(!scope.is_empty());
        assert_eq!(scope.len(), 1);
    }

    #[test]
    fn test_handler_error_propagation() {
        struct FailingService;

        impl Capability for FailingService {
            fn capability_name(&self) -> &'static str {
                "FailingService"
            }
        }

        impl Handles<Echo> for FailingService {
            fn handle(&self, _: Echo) -> Result<String, CapabilityError> {
                Err(CapabilityError::HandlerFailed {
                    message: "intentional failure".to_string(),
                })
            }
        }

        let mut scope = CapabilityScope::new();
        scope.register::<FailingService, Echo>(Arc::new(FailingService));

        let result = scope.dispatch(Echo("test".to_string()));
        assert!(matches!(result, Err(CapabilityError::HandlerFailed { .. })));
    }
}
