# DSPy Examples with GEPA

A collection of practical examples demonstrating how to use [DSPy](https://dspy.ai/).

## About

This repository contains various examples showcasing different applications of DSPy (+ GEPA optimizer).

### What is DSPy?

DSPy is a framework for algorithmically optimizing language model prompts and weights. Instead of manually tweaking prompts, you define what your system should do (the signature), and DSPy figures out how to do it through optimization.

### What is GEPA?

GEPA (Generalized Evolution of Prompting via Adaptation) is a DSPy optimizer that:
- Automatically improves prompts through iterative reflection
- Learns from feedback metrics without requiring labeled data
- Evolves instructions based on performance analysis
- Enables zero-shot learning with just input/output pairs

## Examples

### 1. [PII De-identification](./dspy-gepa-deidentification/)

Demonstrates using GEPA to automatically optimize prompts for redacting personally identifiable information (PII) from incident reports.

**Key concepts:**
- Automatic prompt optimization for sensitive data handling
- Dual metric systems (simple and composite)
- Structured output preservation
- Feedback-driven learning

[View example →](./dspy-gepa-deidentification/)

### 2. [Fact-Checked RAG](./dspy-fact-checker/)

A self-correcting Retrieval-Augmented Generation system that fact-checks its own answers against Wikipedia sources and automatically refines responses until they are fully supported by evidence.

**Key concepts:**
- Self-correcting pipeline with dspy.Refine
- Fact verification against retrieved context
- Wikipedia integration for knowledge retrieval
- Automatic retry and refinement

[View example →](./dspy-fact-checker/)

### 3. [Natural Language to SQL](./dspy-gepa-sql-generator/)

Demonstrates using GEPA to optimize prompts for converting natural language questions into SQL queries, with comprehensive safety and correctness validation.

**Key concepts:**
- Natural language to SQL generation
- Custom metrics for safety, execution, and correctness
- Database schema understanding
- Query optimization through GEPA

[View example →](./dspy-gepa-sql-generator/)

### 4. [Multi-Agent Research Pipeline](./dspy-gepa-researcher/)

An intelligent, multi-agent research pipeline that autonomously conducts web research and generates comprehensive, citation-backed reports using DSPy, LangGraph, and the Exa search API.

**Key concepts:**
- Multi-agent architecture with LangGraph
- Coordinated agents for query planning, search, summarization, writing, and review
- Smart web research powered by Exa API
- Automated writing with proper citations
- Iterative research with gap analysis

[View example →](./dspy-gepa-researcher/)

### 5. [Optimizing Inter-Agent Communication](./dspy-gepa-agent-communication/) ⭐ NEW

A groundbreaking example that optimizes **how agents communicate with each other**, not just individual task performance. Uses DSPy + GEPA + MAST (Multi-Agent System Failure Taxonomy) to discover emergent communication conventions.

**Key concepts:**
- Communication protocols as optimizable DSPy signatures
- MAST taxonomy for identifying where communication breaks down
- Model assertions for real-time quality monitoring
- GEPA optimization of dialogue structure and information exchange
- Emergent communication patterns discovered through optimization

**Why this is novel:**
Unlike traditional approaches that optimize individual agent prompts, this optimizes the *language* agents use to communicate. Metrics measure communication quality (clarity, completeness, routing accuracy) rather than task accuracy. The result: emergent communication conventions that humans wouldn't manually design.

[View example →](./dspy-gepa-agent-communication/)

## Getting Started

Each example directory contains its own README with:
- Detailed setup instructions
- Usage examples
- Configuration details
- Code explanations

Navigate to any example directory to get started.

## Requirements

Most examples require:
- Python 3.13+
- An OpenAI API key (or compatible LLM provider)
- Dependencies managed via [uv](https://github.com/astral-sh/uv) or pip

## Contributing

Additional examples are welcome! If you have a useful DSPy (+ GEPA) example to share:
1. Create a new directory with a descriptive name
2. Include a comprehensive README.md
3. Ensure dependencies are clearly documented
4. Add your example to this README's Examples section
5. Submit a pull request

## Resources

- [DSPy Documentation](https://dspy.ai/)
- [GEPA Optimizer Documentation](https://dspy.ai/api/optimizers/GEPA/overview/)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)

## License

Individual examples may have their own licenses. Please check each example directory for details.

---

**Note**: These are demonstration projects for educational purposes. Always validate and thoroughly test before using in production environments.
