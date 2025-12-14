//! Tool abstraction for LLM function calling.
//!
//! Tools wrap capabilities (via `Handles<R>`) and integrate with the
//! LLM tool-calling interface. This module provides:
//!
//! - `ToolDef`: Type-safe tool definition trait (inspired by rig-core)
//! - `Tool`: A named capability with schema for LLM function calling
//! - `ToolRegistry`: Collection of tools using `CapabilityScope` from core
//!
//! ## Type-Safe Tools with `ToolDef`
//!
//! For compile-time tool names and typed arguments:
//!
//! ```ignore
//! struct Calculator;
//!
//! impl ToolDef for Calculator {
//!     const NAME: &'static str = "calculate";
//!     type Args = CalcArgs;
//!     type Output = String;
//!     type Error = CapabilityError;
//!
//!     fn description() -> &'static str {
//!         "Evaluate a mathematical expression"
//!     }
//!
//!     fn call(args: Self::Args) -> Result<Self::Output, Self::Error> {
//!         // implementation
//!     }
//! }
//! ```

use crate::requests::{ToolInvoke, ToolResult, ToolSchema};
use compositional_core::capability::{CapabilityError, CapabilityScope, Handles};
use serde::{de::DeserializeOwned, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// ToolDef Trait (Type-Safe Tool Definition)
// ============================================================================

/// Type-safe tool definition trait.
///
/// This trait provides compile-time tool names and typed arguments,
/// inspired by rig-core's `Tool` trait.
///
/// ## Example
///
/// ```ignore
/// use compositional_agents::tool::ToolDef;
/// use compositional_core::capability::CapabilityError;
///
/// #[derive(serde::Deserialize)]
/// struct SearchArgs {
///     query: String,
///     limit: Option<usize>,
/// }
///
/// struct WebSearch;
///
/// impl ToolDef for WebSearch {
///     const NAME: &'static str = "web_search";
///     type Args = SearchArgs;
///     type Output = Vec<String>;
///     type Error = CapabilityError;
///
///     fn description() -> &'static str {
///         "Search the web for information"
///     }
///
///     fn call(args: Self::Args) -> Result<Self::Output, Self::Error> {
///         Ok(vec![format!("Results for: {}", args.query)])
///     }
/// }
/// ```
pub trait ToolDef: Send + Sync + 'static {
    /// The tool's unique name (compile-time constant).
    const NAME: &'static str;

    /// Arguments type (must be deserializable from JSON).
    type Args: DeserializeOwned + Send + Sync + 'static;

    /// Output type (must be serializable to JSON).
    type Output: Serialize + Send + Sync + 'static;

    /// Error type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Description of what the tool does.
    fn description() -> &'static str;

    /// Execute the tool with the given arguments.
    fn call(args: Self::Args) -> Result<Self::Output, Self::Error>;

    /// Get the tool's name at runtime.
    fn name() -> &'static str {
        Self::NAME
    }

    /// Build a ToolSchema for this tool.
    ///
    /// Override this to add parameter documentation.
    fn schema() -> ToolSchema {
        ToolSchema::new(Self::NAME, Self::description())
    }

    /// Convert this ToolDef into a Tool instance.
    fn into_tool() -> Tool
    where
        Self: Sized,
    {
        Tool::new(Self::schema(), |args: Self::Args| {
            Self::call(args).map_err(|e| CapabilityError::HandlerFailed {
                message: e.to_string(),
            })
        })
    }
}

// ============================================================================
// Tool Handler Trait
// ============================================================================

/// Type-erased tool handler for dynamic dispatch.
trait AnyToolHandler: Send + Sync {
    /// Handle a tool invocation with JSON arguments.
    fn handle_json(&self, args: serde_json::Value) -> Result<serde_json::Value, CapabilityError>;
}

/// Wrapper that implements AnyToolHandler for any tool function.
struct ToolHandlerWrapper<F, Args, Output>
where
    F: Fn(Args) -> Result<Output, CapabilityError> + Send + Sync,
    Args: DeserializeOwned,
    Output: Serialize,
{
    handler: F,
    _phantom: std::marker::PhantomData<(Args, Output)>,
}

impl<F, Args, Output> AnyToolHandler for ToolHandlerWrapper<F, Args, Output>
where
    F: Fn(Args) -> Result<Output, CapabilityError> + Send + Sync,
    Args: DeserializeOwned + Send + Sync,
    Output: Serialize + Send + Sync,
{
    fn handle_json(&self, args: serde_json::Value) -> Result<serde_json::Value, CapabilityError> {
        let parsed_args: Args =
            serde_json::from_value(args).map_err(|e| CapabilityError::HandlerFailed {
                message: format!("Failed to parse arguments: {}", e),
            })?;

        let result = (self.handler)(parsed_args)?;

        serde_json::to_value(result).map_err(|e| CapabilityError::HandlerFailed {
            message: format!("Failed to serialize result: {}", e),
        })
    }
}

// ============================================================================
// Tool
// ============================================================================

/// A tool that can be invoked by an LLM.
///
/// Tools wrap a handler function and provide a schema for the LLM.
pub struct Tool {
    /// Tool name
    pub name: String,
    /// Tool schema for LLM
    pub schema: ToolSchema,
    /// Type-erased handler
    handler: Box<dyn AnyToolHandler>,
}

impl Tool {
    /// Create a new tool with a handler function.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use compositional_agents::tool::Tool;
    /// use compositional_agents::requests::ToolSchema;
    ///
    /// #[derive(serde::Deserialize)]
    /// struct SearchArgs { query: String }
    ///
    /// let tool = Tool::new(
    ///     ToolSchema::new("search", "Search the web")
    ///         .param("query", "string", "Search query", true),
    ///     |args: SearchArgs| Ok(format!("Results for: {}", args.query)),
    /// );
    /// ```
    pub fn new<F, Args, Output>(schema: ToolSchema, handler: F) -> Self
    where
        F: Fn(Args) -> Result<Output, CapabilityError> + Send + Sync + 'static,
        Args: DeserializeOwned + Send + Sync + 'static,
        Output: Serialize + Send + Sync + 'static,
    {
        Self {
            name: schema.name.clone(),
            schema,
            handler: Box::new(ToolHandlerWrapper {
                handler,
                _phantom: std::marker::PhantomData,
            }),
        }
    }

    /// Invoke the tool with JSON arguments.
    pub fn invoke(&self, args: serde_json::Value) -> Result<serde_json::Value, CapabilityError> {
        self.handler.handle_json(args)
    }
}

impl std::fmt::Debug for Tool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tool")
            .field("name", &self.name)
            .field("schema", &self.schema)
            .finish()
    }
}

// ============================================================================
// Tool Registry
// ============================================================================

/// Registry of tools available to an agent.
///
/// The registry uses `CapabilityScope` from core for capability management
/// and provides tool schemas for LLM function calling.
#[derive(Default)]
pub struct ToolRegistry {
    /// Tools indexed by name
    tools: HashMap<String, Tool>,
    /// Capability scope for Yoneda-style dispatch
    scope: CapabilityScope,
}

impl ToolRegistry {
    /// Create a new empty tool registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            scope: CapabilityScope::new(),
        }
    }

    /// Register a tool.
    pub fn register(&mut self, tool: Tool) {
        self.tools.insert(tool.name.clone(), tool);
    }

    /// Register a tool from a ToolDef type.
    ///
    /// This provides compile-time tool names and typed arguments.
    ///
    /// # Example
    ///
    /// ```ignore
    /// registry.register_def::<Calculator>();
    /// registry.register_def::<WebSearch>();
    /// ```
    pub fn register_def<T: ToolDef>(&mut self) {
        self.register(T::into_tool());
    }

    /// Register a capability that handles `ToolInvoke` requests.
    pub fn register_capability<C>(&mut self, capability: Arc<C>)
    where
        C: Handles<ToolInvoke> + 'static,
    {
        self.scope.register::<C, ToolInvoke>(capability);
    }

    /// Check if a tool exists.
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Get tool schemas for LLM function calling.
    pub fn schemas(&self) -> Vec<ToolSchema> {
        self.tools.values().map(|t| t.schema.clone()).collect()
    }

    /// Get a specific tool's schema.
    pub fn get_schema(&self, name: &str) -> Option<&ToolSchema> {
        self.tools.get(name).map(|t| &t.schema)
    }

    /// Invoke a tool by name.
    pub fn invoke(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<serde_json::Value, CapabilityError> {
        // First try direct tool lookup
        if let Some(tool) = self.tools.get(name) {
            return tool.invoke(args);
        }

        // Fall back to capability scope
        if self.scope.can_handle::<ToolInvoke>() {
            let req = ToolInvoke::new(name, args);
            return self.scope.dispatch(req);
        }

        Err(CapabilityError::NotFound {
            request_type: "Tool",
        })
    }

    /// Invoke a tool and return a ToolResult.
    pub fn invoke_for_result(
        &self,
        tool_call_id: &str,
        name: &str,
        args: serde_json::Value,
    ) -> ToolResult {
        match self.invoke(name, args) {
            Ok(value) => {
                let content = match value {
                    serde_json::Value::String(s) => s,
                    other => other.to_string(),
                };
                ToolResult::success(tool_call_id, content)
            }
            Err(e) => ToolResult::error(tool_call_id, e.to_string()),
        }
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Get tool names.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .finish()
    }
}

// ============================================================================
// Built-in Tools
// ============================================================================

/// Create a calculator tool.
pub fn calculator_tool() -> Tool {
    #[derive(serde::Deserialize)]
    struct CalcArgs {
        expression: String,
    }

    Tool::new(
        ToolSchema::new("calculate", "Evaluate a mathematical expression").param(
            "expression",
            "string",
            "The expression to evaluate",
            true,
        ),
        |args: CalcArgs| {
            // Simple expression evaluator
            let result = evaluate_expression(&args.expression)?;
            Ok(result.to_string())
        },
    )
}

/// Simple expression evaluator (supports +, -, *, /).
fn evaluate_expression(expr: &str) -> Result<f64, CapabilityError> {
    // Very simple parser - just handles "a op b" patterns
    let expr = expr.trim();

    // Try to parse as a single number
    if let Ok(n) = expr.parse::<f64>() {
        return Ok(n);
    }

    // Try to find an operator
    for op in ['+', '-', '*', '/'] {
        if let Some(pos) = expr.rfind(op) {
            if pos > 0 {
                let left = expr[..pos].trim();
                let right = expr[pos + 1..].trim();

                let left_val = evaluate_expression(left)?;
                let right_val = evaluate_expression(right)?;

                return match op {
                    '+' => Ok(left_val + right_val),
                    '-' => Ok(left_val - right_val),
                    '*' => Ok(left_val * right_val),
                    '/' => {
                        if right_val == 0.0 {
                            Err(CapabilityError::HandlerFailed {
                                message: "Division by zero".to_string(),
                            })
                        } else {
                            Ok(left_val / right_val)
                        }
                    }
                    _ => unreachable!(),
                };
            }
        }
    }

    Err(CapabilityError::HandlerFailed {
        message: format!("Cannot parse expression: {}", expr),
    })
}

/// Create a mock search tool.
pub fn mock_search_tool() -> Tool {
    #[derive(serde::Deserialize)]
    struct SearchArgs {
        query: String,
    }

    Tool::new(
        ToolSchema::new("search", "Search the web for information").param(
            "query",
            "string",
            "The search query",
            true,
        ),
        |args: SearchArgs| {
            // Mock search results
            Ok(format!(
                "Search results for '{}': [Result 1: Information about {}] [Result 2: More details on {}]",
                args.query, args.query, args.query
            ))
        },
    )
}

/// Create a mock weather tool.
pub fn mock_weather_tool() -> Tool {
    #[derive(serde::Deserialize)]
    struct WeatherArgs {
        city: String,
    }

    Tool::new(
        ToolSchema::new("weather", "Get current weather for a city").param(
            "city",
            "string",
            "The city name",
            true,
        ),
        |args: WeatherArgs| {
            // Mock weather data
            Ok(format!(
                "Weather in {}: 72°F (22°C), Partly cloudy, Humidity: 45%",
                args.city
            ))
        },
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_creation() {
        #[derive(serde::Deserialize)]
        struct Args {
            name: String,
        }

        let tool = Tool::new(
            ToolSchema::new("greet", "Greet someone").param("name", "string", "Name", true),
            |args: Args| Ok(format!("Hello, {}!", args.name)),
        );

        assert_eq!(tool.name, "greet");
    }

    #[test]
    fn test_tool_invocation() {
        #[derive(serde::Deserialize)]
        struct Args {
            x: i32,
            y: i32,
        }

        let tool = Tool::new(
            ToolSchema::new("add", "Add two numbers")
                .param("x", "number", "First number", true)
                .param("y", "number", "Second number", true),
            |args: Args| Ok(args.x + args.y),
        );

        let result = tool.invoke(serde_json::json!({"x": 2, "y": 3})).unwrap();
        assert_eq!(result, serde_json::json!(5));
    }

    #[test]
    fn test_tool_registry() {
        let mut registry = ToolRegistry::new();
        registry.register(calculator_tool());
        registry.register(mock_search_tool());

        assert!(registry.has_tool("calculate"));
        assert!(registry.has_tool("search"));
        assert!(!registry.has_tool("nonexistent"));

        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn test_registry_invoke() {
        let mut registry = ToolRegistry::new();
        registry.register(calculator_tool());

        let result = registry
            .invoke("calculate", serde_json::json!({"expression": "2 + 3"}))
            .unwrap();
        assert_eq!(result, serde_json::json!("5"));
    }

    #[test]
    fn test_calculator_tool() {
        let tool = calculator_tool();

        let cases = vec![
            ("2 + 3", "5"),
            ("10 - 4", "6"),
            ("3 * 4", "12"),
            ("15 / 3", "5"),
            ("2 + 3 * 4", "14"), // 2 + (3*4) = 14 due to right-to-left parsing
        ];

        for (expr, expected) in cases {
            let result = tool
                .invoke(serde_json::json!({"expression": expr}))
                .unwrap();
            assert_eq!(result, serde_json::json!(expected), "Failed for: {}", expr);
        }
    }

    #[test]
    fn test_invoke_for_result() {
        let mut registry = ToolRegistry::new();
        registry.register(calculator_tool());

        let result = registry.invoke_for_result(
            "call_1",
            "calculate",
            serde_json::json!({"expression": "5 + 5"}),
        );
        assert!(result.success);
        assert_eq!(result.content, "10");
        assert_eq!(result.tool_call_id, "call_1");
    }

    #[test]
    fn test_invoke_nonexistent_tool() {
        let registry = ToolRegistry::new();
        let result = registry.invoke("nonexistent", serde_json::json!({}));
        assert!(result.is_err());
    }

    #[test]
    fn test_schemas() {
        let mut registry = ToolRegistry::new();
        registry.register(calculator_tool());
        registry.register(mock_search_tool());

        let schemas = registry.schemas();
        assert_eq!(schemas.len(), 2);

        let names: Vec<_> = schemas.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"calculate"));
        assert!(names.contains(&"search"));
    }
}
