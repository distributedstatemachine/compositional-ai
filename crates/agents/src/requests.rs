//! Agent-specific request types.
//!
//! These implement the `Request` trait from core, providing agent-specific
//! operations like LLM completion and tool invocation.

use compositional_core::capability::Request;
use serde::{Deserialize, Serialize};

// ============================================================================
// Message Types
// ============================================================================

/// Role in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System message (instructions)
    System,
    /// User message
    User,
    /// Assistant message
    Assistant,
    /// Tool result message
    Tool,
}

/// A message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role of the message sender
    pub role: Role,
    /// Content of the message
    pub content: String,
    /// Optional tool call ID (for tool results)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            tool_call_id: None,
        }
    }

    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            tool_call_id: None,
        }
    }

    /// Create an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            tool_call_id: None,
        }
    }

    /// Create a tool result message.
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

// ============================================================================
// Tool Schema Types
// ============================================================================

/// Schema for a tool parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSchema {
    /// Parameter name
    pub name: String,
    /// Parameter type (e.g., "string", "number", "boolean")
    #[serde(rename = "type")]
    pub param_type: String,
    /// Description of the parameter
    pub description: String,
    /// Whether the parameter is required
    #[serde(default)]
    pub required: bool,
}

/// Schema for a tool (for LLM function calling).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: String,
    /// Parameter schemas
    pub parameters: Vec<ParameterSchema>,
}

impl ToolSchema {
    /// Create a new tool schema.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: Vec::new(),
        }
    }

    /// Add a parameter to the schema.
    pub fn param(
        mut self,
        name: impl Into<String>,
        param_type: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        self.parameters.push(ParameterSchema {
            name: name.into(),
            param_type: param_type.into(),
            description: description.into(),
            required,
        });
        self
    }
}

// ============================================================================
// Tool Call Types
// ============================================================================

/// A tool call request from the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRequest {
    /// Unique ID for this tool call
    pub id: String,
    /// Name of the tool to call
    pub name: String,
    /// Arguments as JSON
    pub arguments: serde_json::Value,
}

/// Result of a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// The tool call ID this result is for
    pub tool_call_id: String,
    /// The result content
    pub content: String,
    /// Whether the call succeeded
    pub success: bool,
}

impl ToolResult {
    /// Create a successful tool result.
    pub fn success(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
            success: true,
        }
    }

    /// Create a failed tool result.
    pub fn error(tool_call_id: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            content: error.into(),
            success: false,
        }
    }
}

// ============================================================================
// LLM Request/Response
// ============================================================================

/// Request for LLM completion.
#[derive(Debug, Clone)]
pub struct LlmRequest {
    /// Conversation messages
    pub messages: Vec<Message>,
    /// Available tools (optional)
    pub tools: Option<Vec<ToolSchema>>,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling
    pub temperature: f32,
}

impl LlmRequest {
    /// Create a new LLM request.
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            tools: None,
            max_tokens: 1024,
            temperature: 0.7,
        }
    }

    /// Add tools to the request.
    pub fn with_tools(mut self, tools: Vec<ToolSchema>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Set max tokens.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }
}

/// Response from LLM completion.
#[derive(Debug, Clone)]
pub enum LlmResponse {
    /// Text response (final answer)
    Text(String),
    /// Tool calls requested
    ToolCalls(Vec<ToolCallRequest>),
}

impl LlmResponse {
    /// Check if this is a final text response.
    pub fn is_text(&self) -> bool {
        matches!(self, LlmResponse::Text(_))
    }

    /// Check if this contains tool calls.
    pub fn is_tool_calls(&self) -> bool {
        matches!(self, LlmResponse::ToolCalls(_))
    }

    /// Get text content if this is a text response.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            LlmResponse::Text(s) => Some(s),
            _ => None,
        }
    }

    /// Get tool calls if this is a tool call response.
    pub fn as_tool_calls(&self) -> Option<&[ToolCallRequest]> {
        match self {
            LlmResponse::ToolCalls(calls) => Some(calls),
            _ => None,
        }
    }
}

impl Request for LlmRequest {
    type Response = LlmResponse;
    fn name() -> &'static str {
        "LlmRequest"
    }
}

// ============================================================================
// Tool Invoke Request
// ============================================================================

/// Request to invoke a tool by name.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInvoke {
    /// Name of the tool to invoke
    pub tool_name: String,
    /// Arguments as JSON
    pub arguments: serde_json::Value,
}

impl ToolInvoke {
    /// Create a new tool invocation request.
    pub fn new(tool_name: impl Into<String>, arguments: serde_json::Value) -> Self {
        Self {
            tool_name: tool_name.into(),
            arguments,
        }
    }
}

impl Request for ToolInvoke {
    type Response = serde_json::Value;
    fn name() -> &'static str {
        "ToolInvoke"
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let sys = Message::system("You are helpful");
        assert_eq!(sys.role, Role::System);
        assert_eq!(sys.content, "You are helpful");

        let user = Message::user("Hello");
        assert_eq!(user.role, Role::User);

        let tool = Message::tool_result("call_123", "Result");
        assert_eq!(tool.role, Role::Tool);
        assert_eq!(tool.tool_call_id, Some("call_123".to_string()));
    }

    #[test]
    fn test_tool_schema_builder() {
        let schema = ToolSchema::new("search", "Search the web")
            .param("query", "string", "The search query", true)
            .param("limit", "number", "Max results", false);

        assert_eq!(schema.name, "search");
        assert_eq!(schema.parameters.len(), 2);
        assert!(schema.parameters[0].required);
        assert!(!schema.parameters[1].required);
    }

    #[test]
    fn test_llm_request_builder() {
        let req = LlmRequest::new(vec![Message::user("Hello")])
            .with_max_tokens(2048)
            .with_temperature(0.5);

        assert_eq!(req.max_tokens, 2048);
        assert!((req.temperature - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_llm_response_variants() {
        let text = LlmResponse::Text("Hello".to_string());
        assert!(text.is_text());
        assert_eq!(text.as_text(), Some("Hello"));

        let calls = LlmResponse::ToolCalls(vec![ToolCallRequest {
            id: "1".to_string(),
            name: "search".to_string(),
            arguments: serde_json::json!({"query": "rust"}),
        }]);
        assert!(calls.is_tool_calls());
        assert_eq!(calls.as_tool_calls().unwrap().len(), 1);
    }

    #[test]
    fn test_tool_result() {
        let success = ToolResult::success("call_1", "Found 10 results");
        assert!(success.success);

        let error = ToolResult::error("call_2", "Network error");
        assert!(!error.success);
    }
}
