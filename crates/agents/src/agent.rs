//! Core agent loop implementation.
//!
//! The `AgentLoop` implements the `Computation` trait from core,
//! providing a traceable async computation that orchestrates
//! LLM calls and tool execution.

use crate::llm::LlmClient;
use crate::requests::{LlmRequest, LlmResponse, Message};
use crate::tool::ToolRegistry;
use crate::trace::{AgentOp, AgentTrace, TraceEvent};
use compositional_core::error::CoreError;
use compositional_core::tracing::Computation;
use std::sync::Arc;
use std::time::Instant;

// ============================================================================
// Agent Configuration
// ============================================================================

/// Configuration for an agent.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Maximum number of tool call iterations
    pub max_iterations: usize,
    /// System prompt
    pub system_prompt: Option<String>,
    /// Maximum tokens for LLM responses
    pub max_tokens: usize,
    /// Temperature for LLM sampling
    pub temperature: f32,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            system_prompt: None,
            max_tokens: 1024,
            temperature: 0.7,
        }
    }
}

impl AgentConfig {
    /// Create a new agent config with a system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set maximum iterations.
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Set max tokens.
    pub fn with_max_tokens(mut self, tokens: usize) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// Set temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }
}

// ============================================================================
// Agent Task and Result
// ============================================================================

/// Input to an agent.
#[derive(Debug, Clone)]
pub struct AgentTask {
    /// The user's request/prompt
    pub prompt: String,
    /// Optional conversation history
    pub history: Vec<Message>,
}

impl AgentTask {
    /// Create a new agent task.
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            history: Vec::new(),
        }
    }

    /// Add conversation history.
    pub fn with_history(mut self, history: Vec<Message>) -> Self {
        self.history = history;
        self
    }
}

/// Result from an agent execution.
#[derive(Debug, Clone)]
pub struct AgentResult {
    /// The final response
    pub response: String,
    /// Number of iterations used
    pub iterations: usize,
    /// Tool calls made
    pub tool_calls: Vec<String>,
    /// Execution trace
    pub trace: AgentTrace,
}

// ============================================================================
// Agent Error
// ============================================================================

/// Errors that can occur during agent execution.
#[derive(Debug, Clone)]
pub enum AgentError {
    /// LLM call failed
    LlmError(String),
    /// Tool execution failed
    ToolError(String),
    /// Maximum iterations exceeded
    MaxIterationsExceeded,
    /// No response from LLM
    NoResponse,
}

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentError::LlmError(msg) => write!(f, "LLM error: {}", msg),
            AgentError::ToolError(msg) => write!(f, "Tool error: {}", msg),
            AgentError::MaxIterationsExceeded => write!(f, "Maximum iterations exceeded"),
            AgentError::NoResponse => write!(f, "No response from LLM"),
        }
    }
}

impl std::error::Error for AgentError {}

impl From<AgentError> for CoreError {
    fn from(e: AgentError) -> Self {
        CoreError::ValidationError {
            reason: e.to_string(),
        }
    }
}

// ============================================================================
// Agent Loop
// ============================================================================

/// The core agent loop.
///
/// This implements the standard agent pattern:
/// 1. Send user message to LLM with tool schemas
/// 2. If LLM returns tool calls, execute them
/// 3. Send tool results back to LLM
/// 4. Repeat until LLM returns final text response
///
/// The agent loop implements `Computation` from core, enabling
/// zero-cost tracing via `Traced<AgentLoop, ENABLED>`.
pub struct AgentLoop<L: LlmClient> {
    /// LLM client
    llm: Arc<L>,
    /// Tool registry
    tools: Arc<ToolRegistry>,
    /// Agent configuration
    config: AgentConfig,
}

impl<L: LlmClient> AgentLoop<L> {
    /// Create a new agent loop.
    pub fn new(llm: Arc<L>, tools: Arc<ToolRegistry>, config: AgentConfig) -> Self {
        Self { llm, tools, config }
    }

    /// Create with default config.
    pub fn with_defaults(llm: Arc<L>, tools: Arc<ToolRegistry>) -> Self {
        Self::new(llm, tools, AgentConfig::default())
    }

    /// Execute the agent loop synchronously.
    pub fn run_sync(&self, task: AgentTask) -> Result<AgentResult, AgentError> {
        let start = Instant::now();
        let mut trace = AgentTrace::new();
        let mut messages = self.build_initial_messages(&task);
        let mut iterations = 0;
        let mut tool_calls_made = Vec::new();

        loop {
            iterations += 1;
            if iterations > self.config.max_iterations {
                return Err(AgentError::MaxIterationsExceeded);
            }

            // Build LLM request
            let tool_schemas = if self.tools.is_empty() {
                None
            } else {
                Some(self.tools.schemas())
            };

            let llm_start = Instant::now();
            let request = LlmRequest::new(messages.clone())
                .with_max_tokens(self.config.max_tokens)
                .with_temperature(self.config.temperature);

            let request = if let Some(schemas) = tool_schemas {
                request.with_tools(schemas)
            } else {
                request
            };

            // Call LLM
            let response = self
                .llm
                .handle(request)
                .map_err(|e| AgentError::LlmError(e.to_string()))?;

            trace.add_event(TraceEvent {
                op: AgentOp::LlmCall {
                    model: self.llm.capability_name().to_string(),
                    input_tokens: messages.iter().map(|m| m.content.len()).sum(),
                    output_tokens: 0, // Would be filled by actual LLM
                },
                duration_ms: llm_start.elapsed().as_millis() as u64,
                success: true,
            });

            match response {
                LlmResponse::Text(text) => {
                    // Final response
                    trace.total_duration_ms = start.elapsed().as_millis() as u64;
                    return Ok(AgentResult {
                        response: text,
                        iterations,
                        tool_calls: tool_calls_made,
                        trace,
                    });
                }
                LlmResponse::ToolCalls(calls) => {
                    // Execute tool calls
                    let mut tool_results = Vec::new();

                    for call in calls {
                        let tool_start = Instant::now();
                        tool_calls_made.push(call.name.clone());

                        let result = self.tools.invoke_for_result(
                            &call.id,
                            &call.name,
                            call.arguments.clone(),
                        );

                        trace.add_event(TraceEvent {
                            op: AgentOp::ToolCall {
                                name: call.name.clone(),
                                args: call.arguments.to_string(),
                            },
                            duration_ms: tool_start.elapsed().as_millis() as u64,
                            success: result.success,
                        });

                        tool_results.push(result);
                    }

                    // Add tool results to messages
                    for result in tool_results {
                        messages.push(Message::tool_result(&result.tool_call_id, &result.content));
                    }
                }
            }
        }
    }

    /// Build initial messages from task.
    fn build_initial_messages(&self, task: &AgentTask) -> Vec<Message> {
        let mut messages = Vec::new();

        // Add system prompt if configured
        if let Some(ref system) = self.config.system_prompt {
            messages.push(Message::system(system));
        }

        // Add history
        messages.extend(task.history.clone());

        // Add user message
        messages.push(Message::user(&task.prompt));

        messages
    }
}

impl<L: LlmClient> Clone for AgentLoop<L> {
    fn clone(&self) -> Self {
        Self {
            llm: Arc::clone(&self.llm),
            tools: Arc::clone(&self.tools),
            config: self.config.clone(),
        }
    }
}

// Implement Computation trait for integration with core's tracing
impl<L: LlmClient + 'static> Computation for AgentLoop<L> {
    type Input = AgentTask;
    type Output = AgentResult;

    fn run(
        &self,
        input: Self::Input,
    ) -> impl std::future::Future<Output = Result<Self::Output, CoreError>> + Send {
        let this = self.clone();
        async move { this.run_sync(input).map_err(CoreError::from) }
    }
}

// ============================================================================
// Simple Agent (convenience wrapper)
// ============================================================================

/// A simple agent that wraps AgentLoop with common defaults.
pub struct SimpleAgent<L: LlmClient> {
    inner: AgentLoop<L>,
}

impl<L: LlmClient> SimpleAgent<L> {
    /// Create a new simple agent.
    pub fn new(llm: Arc<L>) -> Self {
        Self {
            inner: AgentLoop::with_defaults(llm, Arc::new(ToolRegistry::new())),
        }
    }

    /// Create with tools.
    pub fn with_tools(llm: Arc<L>, tools: ToolRegistry) -> Self {
        Self {
            inner: AgentLoop::with_defaults(llm, Arc::new(tools)),
        }
    }

    /// Create with full configuration.
    pub fn with_config(llm: Arc<L>, tools: ToolRegistry, config: AgentConfig) -> Self {
        Self {
            inner: AgentLoop::new(llm, Arc::new(tools), config),
        }
    }

    /// Run the agent with a simple prompt.
    pub fn run(&self, prompt: impl Into<String>) -> Result<AgentResult, AgentError> {
        self.inner.run_sync(AgentTask::new(prompt))
    }

    /// Run with a full task.
    pub fn run_task(&self, task: AgentTask) -> Result<AgentResult, AgentError> {
        self.inner.run_sync(task)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::DeterministicMockLlm;
    use crate::tool::{calculator_tool, mock_search_tool};

    #[test]
    fn test_simple_agent_text_response() {
        let llm = Arc::new(DeterministicMockLlm::text_only(
            "Hello, I'm an AI assistant!",
        ));
        let agent = SimpleAgent::new(llm);

        let result = agent.run("Hi there!").unwrap();
        assert_eq!(result.response, "Hello, I'm an AI assistant!");
        assert_eq!(result.iterations, 1);
        assert!(result.tool_calls.is_empty());
    }

    #[test]
    fn test_agent_with_tool_call() {
        let llm = Arc::new(DeterministicMockLlm::tool_then_text(
            "calculate",
            serde_json::json!({"expression": "2 + 2"}),
            "The result is 4",
        ));

        let mut tools = ToolRegistry::new();
        tools.register(calculator_tool());

        let agent = SimpleAgent::with_tools(llm, tools);
        let result = agent.run("What is 2 + 2?").unwrap();

        assert_eq!(result.response, "The result is 4");
        assert_eq!(result.iterations, 2);
        assert_eq!(result.tool_calls, vec!["calculate"]);
    }

    #[test]
    fn test_agent_config() {
        let config = AgentConfig::default()
            .with_system_prompt("You are helpful")
            .with_max_iterations(5)
            .with_max_tokens(2048)
            .with_temperature(0.5);

        assert_eq!(config.system_prompt, Some("You are helpful".to_string()));
        assert_eq!(config.max_iterations, 5);
        assert_eq!(config.max_tokens, 2048);
        assert!((config.temperature - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_agent_task_with_history() {
        let task = AgentTask::new("What's next?").with_history(vec![
            Message::user("Hello"),
            Message::assistant("Hi there!"),
        ]);

        assert_eq!(task.prompt, "What's next?");
        assert_eq!(task.history.len(), 2);
    }

    #[test]
    fn test_max_iterations() {
        // Create an LLM that always returns tool calls
        let responses: Vec<LlmResponse> = (0..15)
            .map(|i| {
                LlmResponse::ToolCalls(vec![crate::requests::ToolCallRequest {
                    id: format!("call_{}", i),
                    name: "calculate".to_string(),
                    arguments: serde_json::json!({"expression": "1+1"}),
                }])
            })
            .collect();

        let llm = Arc::new(DeterministicMockLlm::new(responses));

        let mut tools = ToolRegistry::new();
        tools.register(calculator_tool());

        let config = AgentConfig::default().with_max_iterations(3);
        let agent = SimpleAgent::with_config(llm, tools, config);

        let result = agent.run("Loop forever");
        assert!(matches!(result, Err(AgentError::MaxIterationsExceeded)));
    }

    #[test]
    fn test_trace_recording() {
        let llm = Arc::new(DeterministicMockLlm::tool_then_text(
            "search",
            serde_json::json!({"query": "rust"}),
            "Found results",
        ));

        let mut tools = ToolRegistry::new();
        tools.register(mock_search_tool());

        let agent = SimpleAgent::with_tools(llm, tools);
        let result = agent.run("Search for rust").unwrap();

        // Should have 2 LLM calls and 1 tool call
        assert!(result.trace.events.len() >= 2);

        // First event should be LLM call
        assert!(matches!(result.trace.events[0].op, AgentOp::LlmCall { .. }));

        // Second event should be tool call
        assert!(matches!(
            result.trace.events[1].op,
            AgentOp::ToolCall { .. }
        ));
    }
}
