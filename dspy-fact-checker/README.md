# DSPy Fact-Checked RAG

A self-correcting Retrieval-Augmented Generation (RAG) system built with [DSPy](https://github.com/stanfordnlp/dspy) that fact-checks its own answers against Wikipedia sources. The system automatically refines responses until they are fully supported by retrieved evidence.

## Features

- **Self-Correcting Pipeline**: Uses `dspy.Refine` to iteratively improve answers based on verification feedback
- **Fact Verification**: Built-in verifier that checks if answer claims are supported by retrieved context
- **Wikipedia Integration**: Custom retriever that fetches relevant passages from Wikipedia
- **Automatic Retry**: Continues refining answers until all claims are verifiable (up to max attempts)
- **Source Attribution**: Embeds Wikipedia URLs in context for transparency

## How It Works

1. **Retrieve**: Fetches relevant Wikipedia passages based on the question
2. **Generate**: Creates an answer using only information from the retrieved context
3. **Verify**: Checks if the answer contains any unsupported claims
4. **Refine**: If verification fails, automatically regenerates with feedback until the answer is fully supported

The pipeline uses DSPy's `Refine` module with a reward function that scores 1.0 only when the verifier confirms all claims are supported by the context.

## Requirements

- Python 3.13+
- OpenAI API key (uses `gpt-4o-mini` by default)

## Installation

1. Change to the project folder after cloning the repo:
```bash
cd dspy-fact-checker
```

2. Install dependencies using [uv](https://github.com/astral-sh/uv):
```bash
uv sync
```

Or with pip:
```bash
pip install dspy-ai wikipedia
```

3. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"  # Linux/Mac
# OR
set OPENAI_API_KEY=your-api-key-here  # Windows
```

## Usage

Run the example:
```bash
uv run fact_check_rag.py
```

### Example Output

```
Question: When did Apollo 11 land on the Moon, and who were the astronauts involved?
--------------------------------------------------------------------------------
Final Answer:
Apollo 11 landed on the Moon on July 20, 1969. The astronauts involved were
Neil Armstrong, Buzz Aldrin, and Michael Collins, with Armstrong and Aldrin
walking on the lunar surface while Collins remained in orbit.
--------------------------------------------------------------------------------
Unsupported Claims: None
--------------------------------------------------------------------------------
Context used:
[1] Apollo 11: Apollo 11 was the American spaceflight that first landed humans...
[2] Neil Armstrong: Neil Alden Armstrong was an American astronaut...
```

### Customizing Questions

Edit the `question` variable in `fact_check_rag.py`:

```python
question = "Who discovered penicillin and in which year was it first reported?"
# or
question = "When was the first FIFA World Cup held, and where?"
```

### Configuration Options

Adjust the RAG parameters:

```python
program = FactCheckedRAG(
    k_passages=4,      # Number of Wikipedia passages to retrieve
    max_attempts=3     # Maximum refinement iterations
)
```

Customize the Wikipedia retriever:

```python
wiki_rm = WikipediaRetriever(
    max_chars_per_passage=1500,  # Characters per passage
    language="en"                 # Wikipedia language code
)
```

## Architecture

### Components

- **WikipediaRetriever**: Custom retriever that searches Wikipedia and formats results for DSPy
- **GenerateAnswer**: Signature for creating answers strictly from provided context
- **VerifyAnswer**: Signature for identifying unsupported claims in answers
- **FactCheckedRAG**: Main module combining retrieval, generation, and verification with refinement

### Key Technologies

- [DSPy](https://github.com/stanfordnlp/dspy): Framework for programming language models
- [Wikipedia API](https://pypi.org/project/wikipedia/): Python library for Wikipedia data
- OpenAI GPT-4o-mini: Language model for generation and verification

## Advanced Usage

### Using Different LLMs

Replace the OpenAI configuration with other supported DSPy models:

```python
# Anthropic Claude
lm = dspy.LM("anthropic/claude-3-5-sonnet-20241022", api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Local models
lm = dspy.LM("ollama/llama2")
```

### Custom Retrievers

Extend the retriever for other knowledge sources:

```python
class CustomRetriever:
    def __call__(self, query: str, k: int = 8):
        # Implement your retrieval logic
        # Must return list[dotdict] with `.long_text` attribute
        pass
```

## Limitations

- Relies on Wikipedia data quality and coverage
- Answer quality depends on the underlying LLM's capabilities
- May require multiple refinement attempts for complex questions
- English Wikipedia by default (configurable)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built with [DSPy](https://github.com/stanfordnlp/dspy) by Stanford NLP
- Uses the [Wikipedia API](https://pypi.org/project/wikipedia/) for knowledge retrieval
