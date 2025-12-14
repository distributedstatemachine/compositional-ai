//! Compile-Time Agent Safety: Lifetimes as Sandboxes (Session 18.5)
//!
//! This module demonstrates how Rust's type system provides **compile-time safety
//! guarantees** for agents that other languages achieve through runtime sandboxing.
//!
//! # Key Insight
//!
//! The borrow checker is a sandbox — agents literally cannot access what's not in their scope!
//!
//! # Lifetime-Scoped Agents
//!
//! ```text
//! fn run_agent() {
//!     let resources = Resources::new();
//!     {
//!         let agent = Agent::new(&resources, config);
//!         agent.call("task").await;
//!     }  // agent dropped, borrow ends
//!
//!     // resources still valid, agent gone
//!     // No leaks, no dangling refs — COMPILER PROVED IT
//! }
//! ```
//!
//! # Comparison with Runtime Sandboxing
//!
//! | Aspect | Python/JS | Rust |
//! |--------|-----------|------|
//! | Isolation | Runtime sandbox | Compile-time lifetimes |
//! | Thread safety | Locks + hope | `Send`/`Sync` proofs |
//! | Capability control | Runtime checks | Trait bounds |
//! | Overhead | Always present | Zero at runtime |
//! | Guarantees | "Should work" | "Will work or won't compile" |
//!
//! # Capability Traits
//!
//! The standard capability traits ([`HasDatabase`], [`HasLLM`], [`HasFileSystem`],
//! [`HasReadOnlyDatabase`]) are defined in `compositional_core::capability` and
//! re-exported here for convenience.

use std::sync::Arc;

// Re-export core capability traits for convenience
pub use compositional_core::{
    CapabilityError, HasDatabase, HasFileSystem, HasLLM, HasReadOnlyDatabase,
};

// ============================================================================
// Mock Implementations (for demonstration)
// ============================================================================

/// A mock database for demonstration.
#[derive(Debug, Clone)]
pub struct MockDatabase {
    /// Stored data
    data: Vec<(String, String)>,
}

impl MockDatabase {
    /// Create a new mock database.
    pub fn new() -> Self {
        Self {
            data: vec![
                ("users".to_string(), "alice, bob, charlie".to_string()),
                ("products".to_string(), "widget, gadget, thing".to_string()),
                ("orders".to_string(), "order1, order2, order3".to_string()),
            ],
        }
    }

    /// Add data to the mock database.
    pub fn with_data(mut self, key: &str, value: &str) -> Self {
        self.data.push((key.to_string(), value.to_string()));
        self
    }

    /// Query the mock database.
    pub fn query(&self, query: &str) -> Vec<String> {
        self.data
            .iter()
            .filter(|(k, _)| query.contains(k.as_str()))
            .map(|(_, v)| v.clone())
            .collect()
    }
}

impl Default for MockDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// A mock LLM client for demonstration.
#[derive(Debug, Clone)]
pub struct MockLLM {
    /// Model name
    model: String,
}

impl MockLLM {
    /// Create a new mock LLM.
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
        }
    }

    /// Generate a mock completion.
    pub fn complete(&self, prompt: &str) -> String {
        format!(
            "[{}] Response to '{}': This is a mock completion.",
            self.model,
            &prompt[..prompt.len().min(50)]
        )
    }
}

/// A mock file system for demonstration.
#[derive(Debug, Clone)]
pub struct MockFileSystem {
    /// Files in the mock filesystem
    files: Vec<(String, String)>,
}

impl MockFileSystem {
    /// Create a new mock file system.
    pub fn new() -> Self {
        Self {
            files: vec![
                ("/etc/config".to_string(), "setting=value".to_string()),
                ("/data/file.txt".to_string(), "Hello, World!".to_string()),
            ],
        }
    }

    /// Read a file.
    pub fn read(&self, path: &str) -> Option<String> {
        self.files
            .iter()
            .find(|(p, _)| p == path)
            .map(|(_, content)| content.clone())
    }

    /// Write a file.
    pub fn write(&mut self, path: &str, content: &str) {
        if let Some(pos) = self.files.iter().position(|(p, _)| p == path) {
            self.files[pos].1 = content.to_string();
        } else {
            self.files.push((path.to_string(), content.to_string()));
        }
    }
}

impl Default for MockFileSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Scope Types (Resource Containers)
// ============================================================================

/// Full scope with database and LLM access.
///
/// This scope provides both database and LLM capabilities.
#[derive(Debug, Clone)]
pub struct FullScope {
    /// Database
    pub db: MockDatabase,
    /// LLM client
    pub llm: MockLLM,
}

impl FullScope {
    /// Create a new full scope.
    pub fn new(db: MockDatabase, llm: MockLLM) -> Self {
        Self { db, llm }
    }
}

impl HasDatabase for FullScope {
    fn query(&self, query: &str) -> Result<Vec<String>, CapabilityError> {
        Ok(self.db.query(query))
    }
}

impl HasLLM for FullScope {
    fn complete(&self, prompt: &str) -> Result<String, CapabilityError> {
        Ok(self.llm.complete(prompt))
    }
}

/// Minimal scope with only database access.
///
/// Demonstrates principle of least privilege — agent can ONLY access database.
#[derive(Debug, Clone)]
pub struct DatabaseOnlyScope {
    /// Database
    pub db: MockDatabase,
}

impl DatabaseOnlyScope {
    /// Create a new database-only scope.
    pub fn new(db: MockDatabase) -> Self {
        Self { db }
    }
}

impl HasDatabase for DatabaseOnlyScope {
    fn query(&self, query: &str) -> Result<Vec<String>, CapabilityError> {
        Ok(self.db.query(query))
    }
}

impl HasReadOnlyDatabase for DatabaseOnlyScope {
    fn read_query(&self, query: &str) -> Result<Vec<String>, CapabilityError> {
        Ok(self.db.query(query))
    }
}

/// Read-only scope with restricted database access.
///
/// Even more restricted — only read operations allowed.
#[derive(Debug, Clone)]
pub struct ReadOnlyScope {
    /// Database (read-only access)
    db: MockDatabase,
}

impl ReadOnlyScope {
    /// Create a new read-only scope.
    pub fn new(db: MockDatabase) -> Self {
        Self { db }
    }
}

impl HasReadOnlyDatabase for ReadOnlyScope {
    fn read_query(&self, query: &str) -> Result<Vec<String>, CapabilityError> {
        Ok(self.db.query(query))
    }
}

// Note: ReadOnlyScope does NOT implement HasDatabase — only read operations!

/// Scope with file system access.
#[derive(Debug)]
pub struct FileSystemScope {
    /// File system
    fs: std::cell::RefCell<MockFileSystem>,
}

impl FileSystemScope {
    /// Create a new file system scope.
    pub fn new(fs: MockFileSystem) -> Self {
        Self {
            fs: std::cell::RefCell::new(fs),
        }
    }
}

impl HasFileSystem for FileSystemScope {
    fn read_file(&self, path: &str) -> Result<String, CapabilityError> {
        self.fs
            .borrow()
            .read(path)
            .ok_or_else(|| CapabilityError::HandlerFailed {
                message: format!("File not found: {}", path),
            })
    }

    fn write_file(&self, path: &str, content: &str) -> Result<(), CapabilityError> {
        self.fs.borrow_mut().write(path, content);
        Ok(())
    }
}

// ============================================================================
// Lifetime-Scoped Agent
// ============================================================================

/// A lifetime-scoped agent.
///
/// The lifetime `'scope` ensures the agent cannot outlive its resources.
/// This is the key insight: **the borrow checker is a sandbox**!
///
/// # What the Lifetime Guarantees
///
/// 1. Agent cannot outlive its resources
/// 2. Agent cannot store dangling references
/// 3. When scope ends, agent is automatically dropped
/// 4. Resources remain valid after agent is gone
///
/// # Example
///
/// ```
/// use compositional_games::scoped::{ScopedAgent, FullScope, MockDatabase, MockLLM};
///
/// let scope = FullScope::new(MockDatabase::new(), MockLLM::new("gpt-4"));
///
/// {
///     let agent = ScopedAgent::new(&scope, "worker");
///     // agent.scope has lifetime of this block
///     let result = agent.execute("query users");
///     // result is valid
/// }  // agent dropped here
///
/// // scope still usable after agent is gone!
/// let _data = scope.db.query("users");
/// ```
pub struct ScopedAgent<'scope, S> {
    /// Reference to the scope (borrows for lifetime 'scope)
    scope: &'scope S,
    /// Agent name
    name: String,
    /// Execution count (for tracking)
    executions: std::cell::Cell<usize>,
}

impl<'scope, S> ScopedAgent<'scope, S> {
    /// Create a new scoped agent.
    ///
    /// The agent borrows the scope for its entire lifetime.
    pub fn new(scope: &'scope S, name: impl Into<String>) -> Self {
        Self {
            scope,
            name: name.into(),
            executions: std::cell::Cell::new(0),
        }
    }

    /// Get the agent's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the number of executions.
    pub fn execution_count(&self) -> usize {
        self.executions.get()
    }
}

// Implementation for agents with database + LLM access
impl<'scope, S> ScopedAgent<'scope, S>
where
    S: HasDatabase + HasLLM,
{
    /// Execute a task using database and LLM.
    ///
    /// Can use: `self.scope.query()`, `self.scope.complete()`
    pub fn execute(&self, task: &str) -> Result<String, CapabilityError> {
        self.executions.set(self.executions.get() + 1);

        // Query database for context
        let context = self.scope.query(task)?;
        let context_str = context.join(", ");

        // Generate response using LLM
        let prompt = format!("Task: {}. Context: {}", task, context_str);
        let response = self.scope.complete(&prompt)?;

        Ok(format!("[{}] {}", self.name, response))
    }
}

// Implementation for agents with only database access
impl<'scope, S> ScopedAgent<'scope, S>
where
    S: HasDatabase,
{
    /// Execute a database-only query.
    ///
    /// Can use: `self.scope.query()`
    /// Cannot use: LLM (not in scope bounds!)
    pub fn query_only(&self, query: &str) -> Result<Vec<String>, CapabilityError> {
        self.executions.set(self.executions.get() + 1);
        self.scope.query(query)
    }
}

// Implementation for agents with read-only database access
impl<'scope, S> ScopedAgent<'scope, S>
where
    S: HasReadOnlyDatabase,
{
    /// Execute a read-only query.
    ///
    /// Even more restricted — only read operations.
    pub fn read_only_query(&self, query: &str) -> Result<Vec<String>, CapabilityError> {
        self.executions.set(self.executions.get() + 1);
        self.scope.read_query(query)
    }
}

// Implementation for agents with file system access
impl<'scope, S> ScopedAgent<'scope, S>
where
    S: HasFileSystem,
{
    /// Read a file.
    pub fn read_file(&self, path: &str) -> Result<String, CapabilityError> {
        self.executions.set(self.executions.get() + 1);
        self.scope.read_file(path)
    }

    /// Write a file.
    pub fn write_file(&self, path: &str, content: &str) -> Result<(), CapabilityError> {
        self.executions.set(self.executions.get() + 1);
        self.scope.write_file(path, content)
    }
}

// ============================================================================
// Thread-Safe Scope (Send + Sync)
// ============================================================================

/// A thread-safe scope for parallel agent execution.
///
/// `S: Send + Sync` means:
/// - `Send`: Type can be transferred to another thread
/// - `Sync`: Type can be shared between threads via `&T`
///
/// The compiler proves thread safety at compile time!
#[derive(Debug, Clone)]
pub struct ThreadSafeScope {
    /// Database (immutable, safe to share)
    db: MockDatabase,
    /// LLM (immutable, safe to share)
    llm: MockLLM,
}

impl ThreadSafeScope {
    /// Create a new thread-safe scope.
    pub fn new(db: MockDatabase, llm: MockLLM) -> Self {
        Self { db, llm }
    }
}

impl HasDatabase for ThreadSafeScope {
    fn query(&self, query: &str) -> Result<Vec<String>, CapabilityError> {
        Ok(self.db.query(query))
    }
}

impl HasLLM for ThreadSafeScope {
    fn complete(&self, prompt: &str) -> Result<String, CapabilityError> {
        Ok(self.llm.complete(prompt))
    }
}

// ThreadSafeScope is Send + Sync because all its fields are
// This is automatically derived by the compiler!

/// Execute multiple agents in parallel.
///
/// This function requires `S: Send + Sync` — the compiler proves thread safety!
///
/// # What Won't Compile
///
/// ```compile_fail
/// // This won't compile — Rc is not Send!
/// fn bad_parallel() {
///     let shared = std::rc::Rc::new(resources);
///     std::thread::spawn(move || {
///         // ERROR: Rc cannot be sent between threads
///         use_resources(&shared);
///     });
/// }
/// ```
pub fn parallel_execute<S>(
    scope: Arc<S>,
    tasks: Vec<String>,
) -> Vec<Result<String, CapabilityError>>
where
    S: HasDatabase + HasLLM + Send + Sync + 'static,
{
    tasks
        .into_iter()
        .map(|task| {
            let scope = Arc::clone(&scope);
            // In a real implementation, this would spawn actual threads
            // For simplicity, we execute sequentially but demonstrate the types
            let agent = ScopedAgent::new(&*scope, format!("worker-{}", task.len()));
            agent.execute(&task)
        })
        .collect()
}

// ============================================================================
// Owned Agent (for when you need ownership)
// ============================================================================

/// An agent that owns its scope.
///
/// Unlike `ScopedAgent` which borrows, this agent owns its resources.
/// Use when the agent needs to live independently of the scope's original owner.
pub struct OwnedAgent<S> {
    /// Owned scope
    scope: S,
    /// Agent name
    name: String,
}

impl<S> OwnedAgent<S> {
    /// Create a new owned agent.
    pub fn new(scope: S, name: impl Into<String>) -> Self {
        Self {
            scope,
            name: name.into(),
        }
    }

    /// Get the agent's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Consume the agent and return the scope.
    pub fn into_scope(self) -> S {
        self.scope
    }

    /// Get a reference to the scope.
    pub fn scope(&self) -> &S {
        &self.scope
    }
}

impl<S> OwnedAgent<S>
where
    S: HasDatabase + HasLLM,
{
    /// Execute a task.
    pub fn execute(&self, task: &str) -> Result<String, CapabilityError> {
        let context = self.scope.query(task)?;
        let prompt = format!("Task: {}. Context: {}", task, context.join(", "));
        let response = self.scope.complete(&prompt)?;
        Ok(format!("[{}] {}", self.name, response))
    }
}

// ============================================================================
// Agent Pipeline (Composition)
// ============================================================================

/// Result of an agent pipeline stage.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// Output from this stage
    pub output: String,
    /// Stage name
    pub stage: String,
}

/// A pipeline of scoped agents.
///
/// Demonstrates compositional agent design.
pub struct AgentPipeline<'scope, S> {
    /// The shared scope
    scope: &'scope S,
    /// Pipeline stages
    stages: Vec<String>,
}

impl<'scope, S> AgentPipeline<'scope, S>
where
    S: HasDatabase + HasLLM,
{
    /// Create a new agent pipeline.
    pub fn new(scope: &'scope S) -> Self {
        Self {
            scope,
            stages: Vec::new(),
        }
    }

    /// Add a stage to the pipeline.
    pub fn add_stage(mut self, name: impl Into<String>) -> Self {
        self.stages.push(name.into());
        self
    }

    /// Execute the pipeline.
    pub fn execute(&self, initial_input: &str) -> Result<Vec<PipelineResult>, CapabilityError> {
        let mut results = Vec::new();
        let mut current_input = initial_input.to_string();

        for stage_name in &self.stages {
            let agent = ScopedAgent::new(self.scope, stage_name);
            let output = agent.execute(&current_input)?;
            results.push(PipelineResult {
                output: output.clone(),
                stage: stage_name.clone(),
            });
            current_input = output;
        }

        Ok(results)
    }
}

// ============================================================================
// Type-Level Security Demonstration
// ============================================================================

/// Demonstrates that capability restrictions are enforced at compile time.
///
/// This function shows the principle of least privilege in action.
pub mod security_demo {
    use super::*;

    /// An agent that can ONLY read from the database.
    ///
    /// It literally cannot do anything else — the type system prevents it!
    pub fn read_only_agent<S: HasReadOnlyDatabase>(
        scope: &S,
        query: &str,
    ) -> Result<Vec<String>, CapabilityError> {
        let agent = ScopedAgent::new(scope, "read-only");
        agent.read_only_query(query)
    }

    /// An agent with full access.
    ///
    /// Has both database and LLM capabilities.
    pub fn full_access_agent<S: HasDatabase + HasLLM>(
        scope: &S,
        task: &str,
    ) -> Result<String, CapabilityError> {
        let agent = ScopedAgent::new(scope, "full-access");
        agent.execute(task)
    }

    /// Demonstrates what WON'T compile.
    ///
    /// This example shows that trying to use `execute` on a scope that doesn't
    /// implement both `HasDatabase` and `HasLLM` will fail at compile time.
    ///
    /// ```compile_fail
    /// use compositional_games::scoped::{ReadOnlyScope, MockDatabase, ScopedAgent};
    ///
    /// let scope = ReadOnlyScope::new(MockDatabase::new());
    /// let agent = ScopedAgent::new(&scope, "test");
    ///
    /// // This won't compile! ReadOnlyScope doesn't implement HasLLM
    /// agent.execute("task");  // ERROR: trait bound `ReadOnlyScope: HasLLM` is not satisfied
    /// ```
    pub fn _compile_fail_demo() {}
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scoped_agent_basic() {
        let scope = FullScope::new(MockDatabase::new(), MockLLM::new("test-model"));

        let agent = ScopedAgent::new(&scope, "test-agent");

        let result = agent.execute("query users").unwrap();
        assert!(result.contains("[test-agent]"));
        assert_eq!(agent.execution_count(), 1);
    }

    #[test]
    fn test_scoped_agent_lifetime() {
        let scope = FullScope::new(MockDatabase::new(), MockLLM::new("test-model"));

        // Agent in inner scope
        {
            let agent = ScopedAgent::new(&scope, "inner-agent");
            let _result = agent.execute("task").unwrap();
        } // agent dropped here

        // Scope still valid!
        let _data = scope.db.query("users");
    }

    #[test]
    fn test_database_only_scope() {
        let scope = DatabaseOnlyScope::new(MockDatabase::new());

        let agent = ScopedAgent::new(&scope, "db-agent");

        // Can query database
        let result = agent.query_only("users").unwrap();
        assert!(!result.is_empty());

        // Note: agent.execute() would not compile because
        // DatabaseOnlyScope doesn't implement HasLLM!
    }

    #[test]
    fn test_read_only_scope() {
        let scope = ReadOnlyScope::new(MockDatabase::new());

        let agent = ScopedAgent::new(&scope, "readonly-agent");

        // Can only do read queries
        let result = agent.read_only_query("users").unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_file_system_scope() {
        let scope = FileSystemScope::new(MockFileSystem::new());

        let agent = ScopedAgent::new(&scope, "fs-agent");

        // Read existing file
        let content = agent.read_file("/etc/config").unwrap();
        assert!(content.contains("setting"));

        // Write new file
        agent.write_file("/tmp/test.txt", "test content").unwrap();

        // Read it back
        let content = agent.read_file("/tmp/test.txt").unwrap();
        assert_eq!(content, "test content");
    }

    #[test]
    fn test_owned_agent() {
        let scope = FullScope::new(MockDatabase::new(), MockLLM::new("owned-model"));

        let agent = OwnedAgent::new(scope, "owned-agent");

        let result = agent.execute("task").unwrap();
        assert!(result.contains("[owned-agent]"));

        // Can get scope back
        let _scope = agent.into_scope();
    }

    #[test]
    fn test_agent_pipeline() {
        let scope = FullScope::new(MockDatabase::new(), MockLLM::new("pipeline-model"));

        let pipeline = AgentPipeline::new(&scope)
            .add_stage("preprocessor")
            .add_stage("analyzer")
            .add_stage("generator");

        let results = pipeline.execute("initial query").unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].stage, "preprocessor");
        assert_eq!(results[1].stage, "analyzer");
        assert_eq!(results[2].stage, "generator");
    }

    #[test]
    fn test_parallel_execute() {
        let scope = Arc::new(ThreadSafeScope::new(
            MockDatabase::new(),
            MockLLM::new("parallel-model"),
        ));

        let tasks = vec![
            "task1".to_string(),
            "task2".to_string(),
            "task3".to_string(),
        ];

        let results = parallel_execute(scope, tasks);

        assert_eq!(results.len(), 3);
        for result in results {
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_security_demo_functions() {
        let full_scope = FullScope::new(MockDatabase::new(), MockLLM::new("model"));
        let read_scope = ReadOnlyScope::new(MockDatabase::new());

        // Full access works
        let result = security_demo::full_access_agent(&full_scope, "task").unwrap();
        assert!(!result.is_empty());

        // Read-only works
        let result = security_demo::read_only_agent(&read_scope, "users").unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_multiple_agents_same_scope() {
        let scope = FullScope::new(MockDatabase::new(), MockLLM::new("shared-model"));

        // Multiple agents can share the same scope
        let agent1 = ScopedAgent::new(&scope, "agent1");
        let agent2 = ScopedAgent::new(&scope, "agent2");

        let result1 = agent1.execute("query users").unwrap();
        let result2 = agent2.execute("query products").unwrap();

        assert!(result1.contains("[agent1]"));
        assert!(result2.contains("[agent2]"));
    }
}
