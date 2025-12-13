# Compositional AI

A Rust-based course exploring category theory foundations for AI/ML systems.

## What This Is

**This is currently a personal learning project.** A 20-session curriculum I'm working through to build practical Rust infrastructure while learning categorical abstractions. Each session = one commit, with readings, implementation tasks, and tests.

**Topics covered:**
- Diagrams & string diagrams (computation graphs as first-class values)
- Monoidal composition (sequential vs parallel)
- Reverse-mode autodiff (backprop as a functor)
- Probabilistic programming & causal inference
- DisCoCat compositional semantics
- Open games & mechanism design
- Operads for multi-input wiring

## Why This Matters for LLM/Agent Systems

The same abstractions apply directly to modern AI systems:

| Concept | LLM/Agent Application |
|---------|----------------------|
| Diagrams | Agent execution traces, tool orchestration graphs |
| Functors | Sandboxing, remote proxying, type transformations |
| Monoidal composition | Parallel tool calls, concurrent sub-agents |
| Operads | Multi-agent orchestration with arity constraints |
| Open games | Incentive-aligned multi-agent coordination |

Systems like [Agentica](https://www.symbolica.ai/blog/beyond-code-mode-agentica) embody these patterns. This course teaches the underlying math so you can build such systems from first principles.

## Getting Started

```bash
# Run tests
just test

# Run all CI checks locally
just ci

# Check specific session
just session1
```

See `course.md` for the full curriculum (created with GPT 5.2, refined with Claude).

## Roadmap

This is a personal learning repo for now. It may eventually evolve into a [Rustlings](https://github.com/rust-lang/rustlings)-style interactive platform for learning category theory × AI — but no promises.

## License

MIT

---

*This README was AI-generated and human-edited.*
