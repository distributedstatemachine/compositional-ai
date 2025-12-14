//! LLM client abstraction.
//!
//! This module provides LLM client implementations that implement
//! `Handles<LlmRequest>` from the core capability system.

use crate::requests::{LlmRequest, LlmResponse, Role, ToolCallRequest};
use compositional_core::capability::{Capability, CapabilityError, Handles};
use std::sync::atomic::{AtomicUsize, Ordering};

// ============================================================================
// LLM Client Trait
// ============================================================================

/// Trait for LLM clients (convenience trait combining Capability + Handles).
pub trait LlmClient: Capability + Handles<LlmRequest> + Send + Sync {}

impl<T> LlmClient for T where T: Capability + Handles<LlmRequest> + Send + Sync {}

// ============================================================================
// Mock LLM Client (for testing and demos)
// ============================================================================

/// A mock LLM client for testing and demonstrations.
///
/// This client simulates LLM behavior without making actual API calls.
/// It can be configured to return specific responses or tool calls.
#[derive(Debug)]
pub struct MockLlmClient {
    /// Model name for identification
    model: String,
    /// Counter for generating unique IDs
    call_counter: AtomicUsize,
    /// Configured responses (tool name -> response)
    tool_responses: std::collections::HashMap<String, String>,
}

impl MockLlmClient {
    /// Create a new mock LLM client.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            call_counter: AtomicUsize::new(0),
            tool_responses: std::collections::HashMap::new(),
        }
    }

    /// Configure a response for when a specific tool is available.
    pub fn with_tool_response(mut self, tool_name: &str, response: &str) -> Self {
        self.tool_responses
            .insert(tool_name.to_string(), response.to_string());
        self
    }

    /// Generate a unique call ID.
    fn next_call_id(&self) -> String {
        let id = self.call_counter.fetch_add(1, Ordering::SeqCst);
        format!("call_{}", id)
    }

    /// Simulate processing a request.
    fn process_request(&self, req: &LlmRequest) -> LlmResponse {
        // Get the last user message
        let last_user_msg = req
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.content.as_str())
            .unwrap_or("");

        // Check if we have tools and should call one
        if let Some(tools) = &req.tools {
            // Look for keywords that suggest tool use
            for tool in tools {
                let tool_name = &tool.name;

                // Check if user message suggests this tool
                let should_call = match tool_name.as_str() {
                    "search" | "web_search" => {
                        last_user_msg.to_lowercase().contains("search")
                            || last_user_msg.to_lowercase().contains("find")
                            || last_user_msg.to_lowercase().contains("look up")
                    }
                    "calculate" | "calculator" | "math" => {
                        last_user_msg.contains('+')
                            || last_user_msg.contains('-')
                            || last_user_msg.contains('*')
                            || last_user_msg.contains('/')
                            || last_user_msg.to_lowercase().contains("calculate")
                            || last_user_msg.to_lowercase().contains("what is")
                    }
                    "weather" => last_user_msg.to_lowercase().contains("weather"),
                    _ => last_user_msg
                        .to_lowercase()
                        .contains(&tool_name.to_lowercase()),
                };

                if should_call {
                    // Check if the last message was a tool result - if so, generate final answer
                    let has_tool_result = req.messages.iter().any(|m| m.role == Role::Tool);
                    if has_tool_result {
                        // Generate final answer based on tool result
                        if let Some(tool_msg) =
                            req.messages.iter().rev().find(|m| m.role == Role::Tool)
                        {
                            return LlmResponse::Text(format!(
                                "Based on the {} result: {}",
                                tool_name, tool_msg.content
                            ));
                        }
                    }

                    // Generate tool call
                    let arguments = self.generate_tool_arguments(tool_name, last_user_msg);
                    return LlmResponse::ToolCalls(vec![ToolCallRequest {
                        id: self.next_call_id(),
                        name: tool_name.clone(),
                        arguments,
                    }]);
                }
            }
        }

        // Default: generate a text response
        LlmResponse::Text(format!(
            "[{}] I received your message: \"{}\"",
            self.model,
            last_user_msg.chars().take(50).collect::<String>()
        ))
    }

    /// Generate mock arguments for a tool call.
    fn generate_tool_arguments(&self, tool_name: &str, user_msg: &str) -> serde_json::Value {
        match tool_name {
            "search" | "web_search" => {
                // Extract search query from message
                let query = user_msg
                    .replace("search for", "")
                    .replace("find", "")
                    .replace("look up", "")
                    .trim()
                    .to_string();
                serde_json::json!({ "query": query })
            }
            "calculate" | "calculator" | "math" => {
                // Extract expression
                let expr = user_msg
                    .replace("calculate", "")
                    .replace("what is", "")
                    .trim()
                    .to_string();
                serde_json::json!({ "expression": expr })
            }
            "weather" => {
                // Extract city
                let city = user_msg
                    .replace("weather in", "")
                    .replace("weather for", "")
                    .trim()
                    .to_string();
                serde_json::json!({ "city": city })
            }
            _ => serde_json::json!({}),
        }
    }
}

impl Capability for MockLlmClient {
    fn capability_name(&self) -> &'static str {
        "MockLlmClient"
    }
}

impl Handles<LlmRequest> for MockLlmClient {
    fn handle(&self, req: LlmRequest) -> Result<LlmResponse, CapabilityError> {
        Ok(self.process_request(&req))
    }
}

// ============================================================================
// Deterministic Mock (for testing)
// ============================================================================

/// A deterministic mock LLM for testing specific scenarios.
#[derive(Debug)]
pub struct DeterministicMockLlm {
    /// Responses to return in order
    responses: std::sync::Mutex<Vec<LlmResponse>>,
}

impl DeterministicMockLlm {
    /// Create a new deterministic mock with the given responses.
    pub fn new(responses: Vec<LlmResponse>) -> Self {
        Self {
            responses: std::sync::Mutex::new(responses),
        }
    }

    /// Create a mock that always returns text.
    pub fn text_only(text: impl Into<String>) -> Self {
        Self::new(vec![LlmResponse::Text(text.into())])
    }

    /// Create a mock that calls a tool then returns text.
    pub fn tool_then_text(
        tool_name: impl Into<String>,
        tool_args: serde_json::Value,
        final_text: impl Into<String>,
    ) -> Self {
        Self::new(vec![
            LlmResponse::ToolCalls(vec![ToolCallRequest {
                id: "test_call_1".to_string(),
                name: tool_name.into(),
                arguments: tool_args,
            }]),
            LlmResponse::Text(final_text.into()),
        ])
    }
}

impl Capability for DeterministicMockLlm {
    fn capability_name(&self) -> &'static str {
        "DeterministicMockLlm"
    }
}

impl Handles<LlmRequest> for DeterministicMockLlm {
    fn handle(&self, _req: LlmRequest) -> Result<LlmResponse, CapabilityError> {
        let mut responses = self.responses.lock().unwrap();
        if responses.is_empty() {
            Ok(LlmResponse::Text(
                "No more responses configured".to_string(),
            ))
        } else {
            Ok(responses.remove(0))
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::requests::{Message, ToolSchema};

    #[test]
    fn test_mock_llm_text_response() {
        let llm = MockLlmClient::new("test-model");
        let req = LlmRequest::new(vec![Message::user("Hello, how are you?")]);

        let response = llm.handle(req).unwrap();
        assert!(response.is_text());
    }

    #[test]
    fn test_mock_llm_tool_call() {
        let llm = MockLlmClient::new("test-model");
        let req = LlmRequest::new(vec![Message::user("Search for rust programming")])
            .with_tools(vec![ToolSchema::new("search", "Search the web")]);

        let response = llm.handle(req).unwrap();
        assert!(response.is_tool_calls());

        let calls = response.as_tool_calls().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
    }

    #[test]
    fn test_mock_llm_calculator() {
        let llm = MockLlmClient::new("test-model");
        let req = LlmRequest::new(vec![Message::user("Calculate 2 + 2")])
            .with_tools(vec![ToolSchema::new("calculate", "Do math")]);

        let response = llm.handle(req).unwrap();
        assert!(response.is_tool_calls());
    }

    #[test]
    fn test_deterministic_mock() {
        let llm = DeterministicMockLlm::tool_then_text(
            "search",
            serde_json::json!({"query": "test"}),
            "Final answer",
        );

        // First call returns tool call
        let req1 = LlmRequest::new(vec![Message::user("test")]);
        let resp1 = llm.handle(req1).unwrap();
        assert!(resp1.is_tool_calls());

        // Second call returns text
        let req2 = LlmRequest::new(vec![Message::user("test")]);
        let resp2 = llm.handle(req2).unwrap();
        assert!(resp2.is_text());
        assert_eq!(resp2.as_text(), Some("Final answer"));
    }
}
