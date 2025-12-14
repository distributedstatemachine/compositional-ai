//! Agent execution hooks for observability.
//!
//! Hooks allow observing agent execution events without modifying the agent loop.
//! This is inspired by rig-core's PromptHook pattern.
//!
//! ## Events
//!
//! - `on_llm_start`: Before sending a request to the LLM
//! - `on_llm_end`: After receiving a response from the LLM
//! - `on_tool_start`: Before executing a tool
//! - `on_tool_end`: After tool execution completes
//!
//! ## Example
//!
//! ```ignore
//! struct LoggingHook;
//!
//! impl AgentHook for LoggingHook {
//!     fn on_tool_start(&self, name: &str, args: &serde_json::Value) {
//!         println!("Calling tool: {} with {:?}", name, args);
//!     }
//! }
//! ```

use crate::requests::{LlmRequest, LlmResponse, ToolResult};

// ============================================================================
// Agent Hook Trait
// ============================================================================

/// Trait for observing agent execution events.
///
/// All methods have default no-op implementations, so you only need to
/// implement the events you care about.
pub trait AgentHook: Send + Sync {
    /// Called before sending a request to the LLM.
    ///
    /// # Arguments
    /// * `request` - The LLM request about to be sent
    /// * `iteration` - Current iteration number (1-indexed)
    fn on_llm_start(&self, _request: &LlmRequest, _iteration: usize) {}

    /// Called after receiving a response from the LLM.
    ///
    /// # Arguments
    /// * `response` - The LLM response received
    /// * `iteration` - Current iteration number
    fn on_llm_end(&self, _response: &LlmResponse, _iteration: usize) {}

    /// Called before executing a tool.
    ///
    /// # Arguments
    /// * `name` - Name of the tool being called
    /// * `args` - Arguments passed to the tool
    fn on_tool_start(&self, _name: &str, _args: &serde_json::Value) {}

    /// Called after tool execution completes.
    ///
    /// # Arguments
    /// * `name` - Name of the tool that was called
    /// * `result` - The result of the tool execution
    fn on_tool_end(&self, _name: &str, _result: &ToolResult) {}

    /// Called when the agent starts processing a task.
    ///
    /// # Arguments
    /// * `prompt` - The user's prompt
    fn on_agent_start(&self, _prompt: &str) {}

    /// Called when the agent finishes processing.
    ///
    /// # Arguments
    /// * `response` - The final response
    /// * `iterations` - Total number of iterations
    fn on_agent_end(&self, _response: &str, _iterations: usize) {}

    /// Called when an error occurs during execution.
    ///
    /// # Arguments
    /// * `error` - Description of the error
    fn on_error(&self, _error: &str) {}
}

// ============================================================================
// Null Hook (Default)
// ============================================================================

/// A no-op hook implementation for when no observation is needed.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullHook;

impl AgentHook for NullHook {}

// ============================================================================
// Logging Hook
// ============================================================================

/// A hook that logs all events to stdout.
#[derive(Debug, Clone, Copy, Default)]
pub struct LoggingHook {
    /// Whether to include verbose details
    pub verbose: bool,
}

impl LoggingHook {
    /// Create a new logging hook.
    pub fn new() -> Self {
        Self { verbose: false }
    }

    /// Create a verbose logging hook.
    pub fn verbose() -> Self {
        Self { verbose: true }
    }
}

impl AgentHook for LoggingHook {
    fn on_llm_start(&self, request: &LlmRequest, iteration: usize) {
        println!("[Hook] LLM call #{} starting...", iteration);
        if self.verbose {
            println!("       Messages: {}", request.messages.len());
            if let Some(tools) = &request.tools {
                println!(
                    "       Tools: {:?}",
                    tools.iter().map(|t| &t.name).collect::<Vec<_>>()
                );
            }
        }
    }

    fn on_llm_end(&self, response: &LlmResponse, iteration: usize) {
        match response {
            LlmResponse::Text(text) => {
                println!(
                    "[Hook] LLM #{} returned text: {}...",
                    iteration,
                    text.chars().take(50).collect::<String>()
                );
            }
            LlmResponse::ToolCalls(calls) => {
                println!(
                    "[Hook] LLM #{} requested {} tool call(s): {:?}",
                    iteration,
                    calls.len(),
                    calls.iter().map(|c| &c.name).collect::<Vec<_>>()
                );
            }
        }
    }

    fn on_tool_start(&self, name: &str, args: &serde_json::Value) {
        println!("[Hook] Tool '{}' starting...", name);
        if self.verbose {
            println!("       Args: {}", args);
        }
    }

    fn on_tool_end(&self, name: &str, result: &ToolResult) {
        let status = if result.success { "success" } else { "failed" };
        println!("[Hook] Tool '{}' {}", name, status);
        if self.verbose {
            println!(
                "       Result: {}",
                result.content.chars().take(100).collect::<String>()
            );
        }
    }

    fn on_agent_start(&self, prompt: &str) {
        println!(
            "[Hook] Agent starting with prompt: {}...",
            prompt.chars().take(50).collect::<String>()
        );
    }

    fn on_agent_end(&self, _response: &str, iterations: usize) {
        println!("[Hook] Agent finished after {} iteration(s)", iterations);
    }

    fn on_error(&self, error: &str) {
        println!("[Hook] Error: {}", error);
    }
}

// ============================================================================
// Composite Hook
// ============================================================================

/// A hook that delegates to multiple inner hooks.
pub struct CompositeHook {
    hooks: Vec<Box<dyn AgentHook>>,
}

impl CompositeHook {
    /// Create a new composite hook.
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    /// Add a hook to the composite.
    pub fn with<H: AgentHook + 'static>(mut self, hook: H) -> Self {
        self.hooks.push(Box::new(hook));
        self
    }
}

impl Default for CompositeHook {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentHook for CompositeHook {
    fn on_llm_start(&self, request: &LlmRequest, iteration: usize) {
        for hook in &self.hooks {
            hook.on_llm_start(request, iteration);
        }
    }

    fn on_llm_end(&self, response: &LlmResponse, iteration: usize) {
        for hook in &self.hooks {
            hook.on_llm_end(response, iteration);
        }
    }

    fn on_tool_start(&self, name: &str, args: &serde_json::Value) {
        for hook in &self.hooks {
            hook.on_tool_start(name, args);
        }
    }

    fn on_tool_end(&self, name: &str, result: &ToolResult) {
        for hook in &self.hooks {
            hook.on_tool_end(name, result);
        }
    }

    fn on_agent_start(&self, prompt: &str) {
        for hook in &self.hooks {
            hook.on_agent_start(prompt);
        }
    }

    fn on_agent_end(&self, response: &str, iterations: usize) {
        for hook in &self.hooks {
            hook.on_agent_end(response, iterations);
        }
    }

    fn on_error(&self, error: &str) {
        for hook in &self.hooks {
            hook.on_error(error);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct CountingHook {
        llm_starts: Arc<AtomicUsize>,
        tool_starts: Arc<AtomicUsize>,
    }

    impl CountingHook {
        fn new() -> (Self, Arc<AtomicUsize>, Arc<AtomicUsize>) {
            let llm_starts = Arc::new(AtomicUsize::new(0));
            let tool_starts = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    llm_starts: Arc::clone(&llm_starts),
                    tool_starts: Arc::clone(&tool_starts),
                },
                llm_starts,
                tool_starts,
            )
        }
    }

    impl AgentHook for CountingHook {
        fn on_llm_start(&self, _request: &LlmRequest, _iteration: usize) {
            self.llm_starts.fetch_add(1, Ordering::SeqCst);
        }

        fn on_tool_start(&self, _name: &str, _args: &serde_json::Value) {
            self.tool_starts.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn test_null_hook() {
        let hook = NullHook;
        // Should not panic
        hook.on_agent_start("test");
        hook.on_agent_end("done", 1);
    }

    #[test]
    fn test_counting_hook() {
        let (hook, llm_count, tool_count) = CountingHook::new();

        let req = LlmRequest::new(vec![]);
        hook.on_llm_start(&req, 1);
        hook.on_llm_start(&req, 2);
        hook.on_tool_start("test", &serde_json::json!({}));

        assert_eq!(llm_count.load(Ordering::SeqCst), 2);
        assert_eq!(tool_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_composite_hook() {
        let (hook1, llm_count1, _) = CountingHook::new();
        let (hook2, llm_count2, _) = CountingHook::new();

        let composite = CompositeHook::new().with(hook1).with(hook2);

        let req = LlmRequest::new(vec![]);
        composite.on_llm_start(&req, 1);

        assert_eq!(llm_count1.load(Ordering::SeqCst), 1);
        assert_eq!(llm_count2.load(Ordering::SeqCst), 1);
    }
}
