# DSPy GEPA for PII De-identification

A minimal example demonstrating how to use DSPy's GEPA (Generalized Evolution of Prompting via Adaptation) optimizer to automatically improve PII (Personally Identifiable Information) redaction in incident reports.

## Overview

This project showcases how GEPA can optimize prompts for sensitive data de-identification tasks through reflection-based prompt evolution. The system learns to:
- Redact emails, phone numbers, and names using standard placeholders
- Preserve document structure (headers, bullet points)
- Maintain causal relationships and action items
- Avoid fabricating new information

## Features

- **Automatic Prompt Optimization**: GEPA evolves instructions based on feedback metrics
- **Dual Metric System**: Includes both a simple and composite metric for evaluation
- **Structured Output**: Maintains "Root cause:" and "Action items:" sections with bullets
- **Zero-Shot Learning**: No labeled examples required - just input/output pairs
- **Feedback-Driven**: Rich textual feedback guides the optimization process

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies with uv
uv sync

# Or with pip
pip install dspy-ai gepa ipykernel
```

### Requirements

- Python 3.13+
- OpenAI API key (for GPT-4o and GPT-4o-mini)

## Configuration

Create a `.env` file in the project root with your OpenAI API key:

```bash
OPENAI_API_KEY=your-api-key-here
```

**Important**: Never commit your `.env` file or API keys to version control.

## Usage

Run the minimal example:

```python
uv run minimal_gepa_deid.py
```

The script will:
1. Define a de-identification signature and module
2. Configure GEPA with a reflection model
3. Optimize the module on training examples
4. Test on a sample incident report

### Example Output

**Input:**
```
Root cause: Dave Miller called 650-555-0000 to report breach.
Action items:
- email dave@contoso.com
- notify legal
```

**Output:**
```
Root cause: [NAME] called [PHONE] to report breach.
Action items:
- email [EMAIL]
- notify legal
```

## How It Works

1. **Signature Definition**: Specifies what the module should do (not how)
2. **Module Creation**: Uses `ChainOfThought` for reasoning about redactions
3. **Metric with Feedback**: Returns both a score and textual guidance
4. **GEPA Optimization**: Evolves internal instructions through reflection
5. **Inference**: Apply the optimized module to new reports

### Metrics

**Simple Metric (`pii_metric`)**:
- 60% score for zero PII leaks
- 20% for preserving "Root cause:" header
- 20% for preserving "Action items:" header

**Composite Metric (`composite_pii_metric`)**:
- Stricter checks including bullet point formatting
- Hallucination detection (no new PII introduction)
- Penalty-based scoring (1.0 - 0.25 � issues)

## Important Notes

### Data Privacy
- This is a **demonstration project** - do not use with real sensitive data without thorough testing
- Never commit actual incident reports or PII to version control
- All data files (`.csv`, `.json`, etc.) are git-ignored by default

### Model Selection
The example uses:
- **Task LM**: `gpt-4o-mini` (faster, cheaper for execution)
- **Reflection LM**: `gpt-4o` (stronger for meta-reasoning about prompts)

You can adjust these in the code based on your needs and budget.

## Development

### Running in Jupyter
The script can also be run in Jupyter notebooks. The project includes `ipykernel` for this purpose.

### Customization
- Modify regex patterns in `EMAIL`, `PHONE`, `NAME` for your use case
- Adjust scoring weights in the metric functions
- Switch between `pii_metric` and `composite_pii_metric` in the GEPA configuration
- Tune GEPA parameters (currently using `auto="light"` for quick demos)

## GEPA Configuration

```python
gepa = dspy.GEPA(
    metric=pii_metric,
    auto="light",              # or "medium"/"heavy" for more optimization
    reflection_lm=reflect_lm,  # Stronger model for reflection
    track_stats=True,          # Track optimization statistics
    track_best_outputs=True    # Keep best candidates per input
)
```

## References

- [DSPy Documentation](https://dspy.ai/)
- [GEPA Overview](https://dspy.ai/api/optimizers/GEPA/overview/)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Built with [DSPy](https://github.com/stanfordnlp/dspy) by Stanford NLP and [GEPA](https://dspy.ai/api/optimizers/GEPA/) optimizer.

---

**Disclaimer**: This is a demonstration project for educational purposes. Always validate and test thoroughly before using in production environments with sensitive data.
