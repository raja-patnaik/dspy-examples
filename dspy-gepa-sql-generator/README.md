# DSPy Natural Language to SQL with GEPA

A demonstration of using [DSPy](https://github.com/stanfordnlp/dspy) with the GEPA optimizer to automatically improve prompts for converting natural language questions into SQL queries. The system learns to generate safe, correct SQL through iterative reflection and feedback.

## Features

- **Automatic Prompt Optimization**: Uses GEPA to evolve SQL generation instructions
- **Safety Validation**: Blocks DDL/DML operations and enforces SELECT-only queries
- **Execution Verification**: Validates SQL syntax and execution correctness
- **Result Correctness**: Compares query results against expected outputs
- **Heuristic Guidance**: Provides feedback on ordering, limits, aliases, and output shape
- **Zero-Shot Learning**: Improves from input/output pairs without labeled prompts

## How It Works

1. **Schema Description**: Creates a compact representation of database tables with sample data
2. **Query Generation**: Uses DSPy ChainOfThought to generate SQL from natural language
3. **Multi-Level Validation**: Custom metric checks safety, execution, and correctness
4. **GEPA Optimization**: Automatically refines prompts based on performance feedback
5. **Iterative Improvement**: Learns from mistakes to generate better queries

The system uses a comprehensive metric that scores queries on:
- **Safety** (40%): Only SELECT/WITH statements, no forbidden operations
- **Execution** (30%): SQL must run without errors
- **Correctness** (30%): Results must match expected output
- **Heuristic penalties**: Deducted for missing ORDER BY, LIMIT, DISTINCT, etc.

## Requirements

- Python 3.13+
- OpenAI API key (or compatible LLM provider)
- DSPy with GEPA optimizer support

## Installation

1. Change to the project folder after cloning the repo:
```bash
cd dspy-gepa-sql-generator
```

2. Install dependencies using [uv](https://github.com/astral-sh/uv):
```bash
uv sync
```

Or with pip:
```bash
pip install dspy-ai
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
uv run nl2sql_gepa.py
```

The script will:
1. Show baseline performance on development queries
2. Run GEPA optimization on training queries
3. Display post-optimization performance improvements
4. Show before/after comparison on sample queries

### Example Output

```
== Baseline on devset ==

Q: Return the top 3 authors by total copies sold in 2024 (name + total_sold), descending.
SQL:
 SELECT a.name, SUM(s.sold) FROM authors a
 JOIN books b ON b.author_id = a.id
 JOIN sales s ON s.book_id = b.id
 WHERE s.year = 2024
 GROUP BY a.name
Score: 0.600
Feedback:
Use ORDER BY ... DESC for descending/top/most queries.
Add LIMIT 3 as requested (top 3).

...

== PostGEPA on devset ==

Q: Return the top 3 authors by total copies sold in 2024 (name + total_sold), descending.
SQL:
 SELECT a.name, SUM(s.sold) AS total_sold
 FROM authors a
 JOIN books b ON b.author_id = a.id
 JOIN sales s ON s.book_id = b.id
 WHERE s.year = 2024
 GROUP BY a.name
 ORDER BY total_sold DESC
 LIMIT 3;
Score: 1.000
Feedback:
Perfect score. Keep current strategy.
```

## Configuration

### Model Settings

Customize the models used for generation and reflection:

```python
STUDENT_MODEL = os.getenv("DSPY_STUDENT_MODEL", "openai/gpt-4o-mini")
REFLECT_MODEL = os.getenv("DSPY_REFLECT_MODEL", "openai/gpt-4o")
```

Set via environment variables:
```bash
export DSPY_STUDENT_MODEL="openai/gpt-4o-mini"
export DSPY_REFLECT_MODEL="openai/gpt-4o"
```

### Database Schema

The example uses an in-memory SQLite database with:
- **authors**: id, name, country
- **books**: id, title, year, author_id, genre, pages, price
- **sales**: book_id, year, sold

Modify `setup_db()` to use your own schema and data.

### Training Set

The example includes 14 questions ranging from simple to complex:
- Basic filtering and joins
- Aggregations and grouping
- Ordering and limiting
- String matching and case handling
- Percentage calculations

Adjust `questions_and_gold_sql()` to add your own examples.

### GEPA Parameters

Tune optimization settings:

```python
gepa = GEPA(
    metric=sql_metric_scalar,
    auto="medium",              # Optimization depth: "light", "medium", "heavy"
    reflection_lm=reflection_lm, # Model for reflection
    track_stats=True,           # Track optimization statistics
)
```

## Architecture

### Components

**NL2SQL Signature**: Defines the input/output specification for SQL generation
- Input: Database schema with sample rows, natural language question
- Output: Single safe SQL SELECT statement

**NL2SQLProgram Module**: Wraps ChainOfThought for SQL generation

**WikipediaRetriever**: Custom metric combining multiple validation layers:
- Safety checks (forbidden keywords, statement type)
- Execution validation (syntax, runtime errors)
- Correctness verification (result matching)
- Heuristic penalties (ordering, limits, aliases)

**GEPA Optimizer**: Iteratively improves prompts through reflection

### Key Technologies

- [DSPy](https://github.com/stanfordnlp/dspy): Framework for programming language models
- [GEPA](https://dspy.ai/api/optimizers/GEPA/overview/): Generalized Evolution of Prompting via Adaptation
- SQLite: Lightweight database for validation
- OpenAI GPT-4o/GPT-4o-mini: Language models for generation and reflection

## Advanced Usage

### Custom Databases

Replace the in-memory SQLite database with your own:

```python
def setup_db() -> sqlite3.Connection:
    conn = sqlite3.connect("your_database.db")
    return conn
```

### Different LLM Providers

Use other DSPy-supported models:

```python
# Anthropic Claude
student_lm = dspy.LM("anthropic/claude-3-5-sonnet-20241022")

# Local models
student_lm = dspy.LM("ollama/llama3")
```

### Metric Customization

Adjust scoring weights in `sql_metric()`:

```python
# Current weights:
# +0.4 for safety
# +0.3 for execution
# +0.3 for correctness
# -0.05 for each heuristic violation
```

## Limitations

- Requires gold SQL examples for correctness validation (can work without, but with reduced signal)
- Limited to SELECT queries (DML/DDL operations are blocked)
- Performance depends on LLM capabilities and schema complexity
- English language queries by default

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built with [DSPy](https://github.com/stanfordnlp/dspy) by Stanford NLP
- Uses the [GEPA optimizer](https://dspy.ai/api/optimizers/GEPA/overview/) for automatic prompt improvement
