//! Agent execution traces.
//!
//! This module provides types for recording and visualizing agent execution,
//! including conversion to `Diagram` from core for graph-based analysis.

use compositional_core::diagram::{Diagram, Node, Port};
use compositional_core::Shape;
use serde::{Deserialize, Serialize};

// ============================================================================
// Agent Operations
// ============================================================================

/// Operations that can occur during agent execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentOp {
    /// LLM call
    LlmCall {
        /// Model name
        model: String,
        /// Input token count
        input_tokens: usize,
        /// Output token count
        output_tokens: usize,
    },
    /// Tool call
    ToolCall {
        /// Tool name
        name: String,
        /// Arguments (as JSON string)
        args: String,
    },
    /// Decision point
    Decision {
        /// Description of decision
        description: String,
    },
    /// Agent start
    Start {
        /// Task description
        task: String,
    },
    /// Agent end
    End {
        /// Final response
        response: String,
    },
}

impl std::fmt::Display for AgentOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentOp::LlmCall { model, .. } => write!(f, "LLM({})", model),
            AgentOp::ToolCall { name, .. } => write!(f, "Tool({})", name),
            AgentOp::Decision { description } => write!(f, "Decision({})", description),
            AgentOp::Start { task } => {
                write!(f, "Start({})", task.chars().take(20).collect::<String>())
            }
            AgentOp::End { response } => {
                write!(f, "End({})", response.chars().take(20).collect::<String>())
            }
        }
    }
}

// ============================================================================
// Trace Event
// ============================================================================

/// A single event in an agent's execution trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEvent {
    /// The operation that occurred
    pub op: AgentOp,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Whether the operation succeeded
    pub success: bool,
}

impl TraceEvent {
    /// Create a new trace event.
    pub fn new(op: AgentOp, duration_ms: u64, success: bool) -> Self {
        Self {
            op,
            duration_ms,
            success,
        }
    }
}

// ============================================================================
// Agent Trace
// ============================================================================

/// Complete trace of an agent's execution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentTrace {
    /// Sequence of events
    pub events: Vec<TraceEvent>,
    /// Total duration in milliseconds
    pub total_duration_ms: u64,
}

impl AgentTrace {
    /// Create a new empty trace.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an event to the trace.
    pub fn add_event(&mut self, event: TraceEvent) {
        self.events.push(event);
    }

    /// Get the number of LLM calls.
    pub fn llm_call_count(&self) -> usize {
        self.events
            .iter()
            .filter(|e| matches!(e.op, AgentOp::LlmCall { .. }))
            .count()
    }

    /// Get the number of tool calls.
    pub fn tool_call_count(&self) -> usize {
        self.events
            .iter()
            .filter(|e| matches!(e.op, AgentOp::ToolCall { .. }))
            .count()
    }

    /// Get total LLM latency.
    pub fn total_llm_latency_ms(&self) -> u64 {
        self.events
            .iter()
            .filter(|e| matches!(e.op, AgentOp::LlmCall { .. }))
            .map(|e| e.duration_ms)
            .sum()
    }

    /// Get total tool latency.
    pub fn total_tool_latency_ms(&self) -> u64 {
        self.events
            .iter()
            .filter(|e| matches!(e.op, AgentOp::ToolCall { .. }))
            .map(|e| e.duration_ms)
            .sum()
    }

    /// Get tools that were called.
    pub fn tools_called(&self) -> Vec<String> {
        self.events
            .iter()
            .filter_map(|e| match &e.op {
                AgentOp::ToolCall { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect()
    }

    /// Check if all events succeeded.
    pub fn all_succeeded(&self) -> bool {
        self.events.iter().all(|e| e.success)
    }

    /// Convert trace to a string diagram for visualization.
    pub fn to_diagram(&self) -> Diagram<AgentOp> {
        let mut diagram = Diagram::new();

        // Add nodes for each event
        let node_indices: Vec<_> = self
            .events
            .iter()
            .map(|event| {
                let node = Node {
                    op: event.op.clone(),
                    inputs: vec![Port::new(Shape::f32_scalar())],
                    outputs: vec![Port::new(Shape::f32_scalar())],
                };
                diagram.add_node(node)
            })
            .collect();

        // Connect sequential nodes
        for i in 0..node_indices.len().saturating_sub(1) {
            let _ = diagram.connect(node_indices[i], 0, node_indices[i + 1], 0);
        }

        diagram
    }

    /// Render trace as ASCII art.
    pub fn render_ascii(&self) -> String {
        let mut output = String::new();
        output.push_str("┌─────────────────────────────────────────┐\n");
        output.push_str("│           Agent Execution Trace         │\n");
        output.push_str("├─────────────────────────────────────────┤\n");

        for (i, event) in self.events.iter().enumerate() {
            let status = if event.success { "✓" } else { "✗" };
            let op_str = format!("{}", event.op);
            let duration = format!("{}ms", event.duration_ms);

            output.push_str(&format!(
                "│ {} {} {:30} {:>6} │\n",
                i + 1,
                status,
                op_str.chars().take(30).collect::<String>(),
                duration
            ));

            if i < self.events.len() - 1 {
                output.push_str("│     │                                     │\n");
                output.push_str("│     ▼                                     │\n");
            }
        }

        output.push_str("├─────────────────────────────────────────┤\n");
        output.push_str(&format!(
            "│ Total: {}ms  LLM: {}  Tools: {}          │\n",
            self.total_duration_ms,
            self.llm_call_count(),
            self.tool_call_count()
        ));
        output.push_str("└─────────────────────────────────────────┘\n");

        output
    }

    /// Render trace as DOT format for Graphviz.
    pub fn render_dot(&self) -> String {
        let mut output = String::new();
        output.push_str("digraph AgentTrace {\n");
        output.push_str("  rankdir=TB;\n");
        output.push_str("  node [shape=box];\n\n");

        // Add nodes
        for (i, event) in self.events.iter().enumerate() {
            let color = match &event.op {
                AgentOp::LlmCall { .. } => "lightblue",
                AgentOp::ToolCall { .. } => "lightgreen",
                AgentOp::Decision { .. } => "lightyellow",
                AgentOp::Start { .. } => "lightgray",
                AgentOp::End { .. } => "lightgray",
            };

            let label = format!("{}", event.op).replace('"', "'");
            output.push_str(&format!(
                "  n{} [label=\"{}\\n{}ms\" style=filled fillcolor={}];\n",
                i, label, event.duration_ms, color
            ));
        }

        output.push('\n');

        // Add edges
        for i in 0..self.events.len().saturating_sub(1) {
            output.push_str(&format!("  n{} -> n{};\n", i, i + 1));
        }

        output.push_str("}\n");
        output
    }
}

// ============================================================================
// Trace Summary
// ============================================================================

/// Summary statistics for an agent trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSummary {
    /// Total events
    pub total_events: usize,
    /// LLM calls
    pub llm_calls: usize,
    /// Tool calls
    pub tool_calls: usize,
    /// Total duration
    pub total_duration_ms: u64,
    /// LLM latency
    pub llm_latency_ms: u64,
    /// Tool latency
    pub tool_latency_ms: u64,
    /// Success rate
    pub success_rate: f64,
    /// Tools used
    pub tools_used: Vec<String>,
}

impl From<&AgentTrace> for TraceSummary {
    fn from(trace: &AgentTrace) -> Self {
        let total_events = trace.events.len();
        let successful = trace.events.iter().filter(|e| e.success).count();
        let success_rate = if total_events > 0 {
            successful as f64 / total_events as f64
        } else {
            1.0
        };

        Self {
            total_events,
            llm_calls: trace.llm_call_count(),
            tool_calls: trace.tool_call_count(),
            total_duration_ms: trace.total_duration_ms,
            llm_latency_ms: trace.total_llm_latency_ms(),
            tool_latency_ms: trace.total_tool_latency_ms(),
            success_rate,
            tools_used: trace.tools_called(),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_trace() -> AgentTrace {
        let mut trace = AgentTrace::new();
        trace.add_event(TraceEvent::new(
            AgentOp::LlmCall {
                model: "test-model".to_string(),
                input_tokens: 100,
                output_tokens: 50,
            },
            150,
            true,
        ));
        trace.add_event(TraceEvent::new(
            AgentOp::ToolCall {
                name: "search".to_string(),
                args: r#"{"query":"test"}"#.to_string(),
            },
            200,
            true,
        ));
        trace.add_event(TraceEvent::new(
            AgentOp::LlmCall {
                model: "test-model".to_string(),
                input_tokens: 200,
                output_tokens: 100,
            },
            180,
            true,
        ));
        trace.total_duration_ms = 530;
        trace
    }

    #[test]
    fn test_trace_counts() {
        let trace = sample_trace();
        assert_eq!(trace.llm_call_count(), 2);
        assert_eq!(trace.tool_call_count(), 1);
    }

    #[test]
    fn test_trace_latencies() {
        let trace = sample_trace();
        assert_eq!(trace.total_llm_latency_ms(), 330); // 150 + 180
        assert_eq!(trace.total_tool_latency_ms(), 200);
    }

    #[test]
    fn test_tools_called() {
        let trace = sample_trace();
        let tools = trace.tools_called();
        assert_eq!(tools, vec!["search"]);
    }

    #[test]
    fn test_all_succeeded() {
        let trace = sample_trace();
        assert!(trace.all_succeeded());
    }

    #[test]
    fn test_to_diagram() {
        let trace = sample_trace();
        let diagram = trace.to_diagram();
        assert_eq!(diagram.node_count(), 3);
    }

    #[test]
    fn test_render_ascii() {
        let trace = sample_trace();
        let ascii = trace.render_ascii();
        assert!(ascii.contains("Agent Execution Trace"));
        assert!(ascii.contains("LLM"));
        assert!(ascii.contains("Tool"));
    }

    #[test]
    fn test_render_dot() {
        let trace = sample_trace();
        let dot = trace.render_dot();
        assert!(dot.contains("digraph"));
        assert!(dot.contains("n0 -> n1"));
    }

    #[test]
    fn test_trace_summary() {
        let trace = sample_trace();
        let summary = TraceSummary::from(&trace);
        assert_eq!(summary.total_events, 3);
        assert_eq!(summary.llm_calls, 2);
        assert_eq!(summary.tool_calls, 1);
        assert!((summary.success_rate - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_agent_op_display() {
        let op = AgentOp::LlmCall {
            model: "gpt-4".to_string(),
            input_tokens: 100,
            output_tokens: 50,
        };
        assert_eq!(format!("{}", op), "LLM(gpt-4)");

        let op = AgentOp::ToolCall {
            name: "search".to_string(),
            args: "{}".to_string(),
        };
        assert_eq!(format!("{}", op), "Tool(search)");
    }
}
