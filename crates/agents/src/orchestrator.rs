//! Multi-agent orchestration using operads.
//!
//! This module provides `Orchestrator` which uses `WiringPlan` from core
//! to validate multi-agent pipelines with arity enforcement.

use crate::agent::{AgentError, AgentResult, AgentTask};
use crate::llm::LlmClient;
use crate::trace::{AgentOp, AgentTrace, TraceEvent};
use compositional_core::operad::{OperadError, Operation, WiringPlan};
use std::sync::Arc;
use std::time::Instant;

// ============================================================================
// Agent Runner Trait
// ============================================================================

/// Trait for something that can run agent tasks.
pub trait AgentRunner: Send + Sync {
    /// Run an agent task.
    fn run(&self, task: AgentTask) -> Result<AgentResult, AgentError>;

    /// Get the agent's name.
    fn name(&self) -> &str;

    /// Get the number of expected inputs (arity).
    fn arity(&self) -> usize;
}

// ============================================================================
// Agent Wrapper
// ============================================================================

/// Wrapper that makes an AgentLoop implement AgentRunner.
pub struct AgentWrapper<L: LlmClient> {
    /// Agent name
    name: String,
    /// The agent loop
    agent: crate::agent::AgentLoop<L>,
    /// Number of tool outputs this agent expects
    arity: usize,
}

impl<L: LlmClient> AgentWrapper<L> {
    /// Create a new agent wrapper.
    pub fn new(name: impl Into<String>, agent: crate::agent::AgentLoop<L>, arity: usize) -> Self {
        Self {
            name: name.into(),
            agent,
            arity,
        }
    }
}

impl<L: LlmClient + 'static> AgentRunner for AgentWrapper<L> {
    fn run(&self, task: AgentTask) -> Result<AgentResult, AgentError> {
        self.agent.run_sync(task)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn arity(&self) -> usize {
        self.arity
    }
}

// ============================================================================
// Pipeline Stage
// ============================================================================

/// A stage in a multi-agent pipeline.
pub struct PipelineStage {
    /// Stage name
    pub name: String,
    /// Agent runner for this stage
    pub agent: Arc<dyn AgentRunner>,
    /// Input indices (which previous stages feed into this one)
    pub inputs: Vec<usize>,
}

impl PipelineStage {
    /// Create a new pipeline stage.
    pub fn new(name: impl Into<String>, agent: Arc<dyn AgentRunner>) -> Self {
        Self {
            name: name.into(),
            agent,
            inputs: Vec::new(),
        }
    }

    /// Set input stage indices.
    pub fn with_inputs(mut self, inputs: Vec<usize>) -> Self {
        self.inputs = inputs;
        self
    }
}

// ============================================================================
// Orchestrator
// ============================================================================

/// Multi-agent orchestrator using operadic composition.
///
/// The orchestrator validates that agents are wired correctly
/// (right number of inputs) before execution using `WiringPlan` from core.
pub struct Orchestrator {
    /// Pipeline stages
    stages: Vec<PipelineStage>,
}

impl Orchestrator {
    /// Create a new orchestrator.
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Add a stage to the pipeline.
    pub fn add_stage(&mut self, stage: PipelineStage) {
        self.stages.push(stage);
    }

    /// Builder method to add a stage.
    pub fn with_stage(mut self, stage: PipelineStage) -> Self {
        self.add_stage(stage);
        self
    }

    /// Validate the pipeline using operadic constraints.
    pub fn validate(&self) -> Result<(), OperadError> {
        if self.stages.is_empty() {
            return Ok(());
        }

        // For each stage, check that inputs are valid
        for (i, stage) in self.stages.iter().enumerate() {
            // Check input indices are valid
            for &input_idx in &stage.inputs {
                if input_idx >= i {
                    return Err(OperadError::InvalidOperation { index: input_idx });
                }
            }

            // Check arity matches
            let expected_arity = stage.agent.arity();
            if expected_arity > 0 && stage.inputs.len() != expected_arity {
                return Err(OperadError::ArityMismatch {
                    operation: stage.name.clone(),
                    expected: expected_arity,
                    got: stage.inputs.len(),
                });
            }
        }

        Ok(())
    }

    /// Create a WiringPlan representation for analysis.
    pub fn to_wiring_plan(&self) -> Option<WiringPlan> {
        if self.stages.is_empty() {
            return None;
        }

        // Create operations for each stage
        let operations: Vec<Operation> = self
            .stages
            .iter()
            .map(|s| Operation::new(&s.name, s.agent.arity()))
            .collect();

        // Last stage is the outer operation
        let outer = operations.last()?.clone();

        // All but last are inner operations
        let inner: Vec<_> = operations.into_iter().take(self.stages.len() - 1).collect();

        // Build wiring from input indices
        let wiring: Vec<(usize, usize)> = self
            .stages
            .last()?
            .inputs
            .iter()
            .enumerate()
            .map(|(slot, &inner_idx)| (inner_idx, slot))
            .collect();

        Some(WiringPlan::new(outer).with_inner(inner).with_wiring(wiring))
    }

    /// Execute the pipeline.
    pub fn execute(&self, initial_input: &str) -> Result<OrchestratorResult, AgentError> {
        self.validate()
            .map_err(|e| AgentError::ToolError(e.to_string()))?;

        let start = Instant::now();
        let mut trace = AgentTrace::new();
        let mut results: Vec<AgentResult> = Vec::new();

        for (i, stage) in self.stages.iter().enumerate() {
            let stage_start = Instant::now();

            // Build input for this stage
            let input = if stage.inputs.is_empty() {
                // First stage or no dependencies: use initial input
                initial_input.to_string()
            } else {
                // Combine outputs from input stages
                stage
                    .inputs
                    .iter()
                    .map(|&idx| results[idx].response.clone())
                    .collect::<Vec<_>>()
                    .join("\n\n")
            };

            // Run the agent
            let task = AgentTask::new(input);
            let result = stage.agent.run(task)?;

            trace.add_event(TraceEvent::new(
                AgentOp::Decision {
                    description: format!("Stage {}: {}", i, stage.name),
                },
                stage_start.elapsed().as_millis() as u64,
                true,
            ));

            // Merge stage trace
            for event in &result.trace.events {
                trace.add_event(event.clone());
            }

            results.push(result);
        }

        trace.total_duration_ms = start.elapsed().as_millis() as u64;

        let final_response = results
            .last()
            .map(|r| r.response.clone())
            .unwrap_or_default();

        Ok(OrchestratorResult {
            response: final_response,
            stage_results: results,
            trace,
        })
    }

    /// Number of stages.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
}

impl Default for Orchestrator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Orchestrator Result
// ============================================================================

/// Result from orchestrator execution.
#[derive(Debug, Clone)]
pub struct OrchestratorResult {
    /// Final response from the last stage
    pub response: String,
    /// Results from each stage
    pub stage_results: Vec<AgentResult>,
    /// Combined trace
    pub trace: AgentTrace,
}

// ============================================================================
// Sequential Pipeline (convenience)
// ============================================================================

/// Create a simple sequential pipeline where each agent passes output to the next.
pub fn sequential_pipeline(agents: Vec<Arc<dyn AgentRunner>>) -> Orchestrator {
    let mut orchestrator = Orchestrator::new();

    for (i, agent) in agents.into_iter().enumerate() {
        let inputs = if i == 0 { vec![] } else { vec![i - 1] };
        orchestrator
            .add_stage(PipelineStage::new(agent.name().to_string(), agent).with_inputs(inputs));
    }

    orchestrator
}

// ============================================================================
// Fan-Out Pipeline
// ============================================================================

/// Create a fan-out pipeline where multiple agents process the same input.
pub fn fanout_pipeline(agents: Vec<Arc<dyn AgentRunner>>) -> Orchestrator {
    let mut orchestrator = Orchestrator::new();

    // All agents run in parallel on the initial input (no dependencies)
    for agent in agents {
        orchestrator.add_stage(PipelineStage::new(agent.name().to_string(), agent));
    }

    orchestrator
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{AgentConfig, AgentLoop};
    use crate::llm::DeterministicMockLlm;
    use crate::tool::ToolRegistry;
    use std::sync::Arc;

    fn mock_agent(name: &str, response: &str) -> Arc<dyn AgentRunner> {
        let llm = Arc::new(DeterministicMockLlm::text_only(response));
        let tools = Arc::new(ToolRegistry::new());
        let agent = AgentLoop::new(llm, tools, AgentConfig::default());
        Arc::new(AgentWrapper::new(name, agent, 0))
    }

    #[test]
    fn test_empty_orchestrator() {
        let orchestrator = Orchestrator::new();
        assert!(orchestrator.validate().is_ok());
    }

    #[test]
    fn test_single_stage() {
        let agent = mock_agent("analyzer", "Analysis complete");
        let orchestrator = Orchestrator::new().with_stage(PipelineStage::new("analyze", agent));

        assert!(orchestrator.validate().is_ok());
        assert_eq!(orchestrator.stage_count(), 1);
    }

    #[test]
    fn test_sequential_execution() {
        let agents = vec![
            mock_agent("stage1", "Stage 1 output"),
            mock_agent("stage2", "Stage 2 output"),
            mock_agent("stage3", "Final output"),
        ];

        let orchestrator = sequential_pipeline(agents);
        assert!(orchestrator.validate().is_ok());

        let result = orchestrator.execute("Initial input").unwrap();
        assert_eq!(result.response, "Final output");
        assert_eq!(result.stage_results.len(), 3);
    }

    #[test]
    fn test_fanout_execution() {
        let agents = vec![
            mock_agent("researcher", "Research results"),
            mock_agent("analyst", "Analysis results"),
            mock_agent("writer", "Written output"),
        ];

        let orchestrator = fanout_pipeline(agents);
        assert!(orchestrator.validate().is_ok());

        let result = orchestrator.execute("Topic to explore").unwrap();
        assert_eq!(result.stage_results.len(), 3);
    }

    #[test]
    fn test_invalid_input_index() {
        let agent = mock_agent("test", "output");
        let mut orchestrator = Orchestrator::new();
        orchestrator.add_stage(
            PipelineStage::new("stage1", agent.clone()).with_inputs(vec![5]), // Invalid!
        );

        let result = orchestrator.validate();
        assert!(matches!(result, Err(OperadError::InvalidOperation { .. })));
    }

    #[test]
    fn test_to_wiring_plan() {
        let agents = vec![
            mock_agent("tool1", "output1"),
            mock_agent("tool2", "output2"),
            mock_agent("combiner", "combined"),
        ];

        let mut orchestrator = Orchestrator::new();
        orchestrator.add_stage(PipelineStage::new("tool1", agents[0].clone()));
        orchestrator.add_stage(PipelineStage::new("tool2", agents[1].clone()));
        orchestrator
            .add_stage(PipelineStage::new("combiner", agents[2].clone()).with_inputs(vec![0, 1]));

        let plan = orchestrator.to_wiring_plan();
        assert!(plan.is_some());

        let plan = plan.unwrap();
        assert_eq!(plan.outer.name, "combiner");
        assert_eq!(plan.inner.len(), 2);
    }
}
