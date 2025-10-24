# DSPy Multi-Agent Research Pipeline

An intelligent, multi-agent research pipeline that autonomously conducts web research and generates comprehensive, citation-backed reports using DSPy, LangGraph, and the Exa search API.

## Features

- **> Multi-Agent Architecture**: Coordinated agents for query planning, search, summarization, writing, and review using LangGraph
- **=
 Smart Web Research**: Powered by Exa API for high-quality web search with full-text retrieval
- **=� Automated Writing**: Generates polished Markdown reports with proper citations and references
- **<� Prompt Optimization**: Optional GEPA (Generalized Efficient Prompt Adaptation) for automatic prompt tuning
- **= Iterative Research**: Gap analysis determines when additional research is needed
- ** Quality Assurance**: Built-in review and revision cycle for report quality
- **=� Structured Citations**: Global citation management with numbered references

## Architecture

The pipeline uses a graph-based workflow with the following stages:

1. **Query Planning** � Generate diverse search queries with operators (site:, intitle:, quoted phrases)
2. **Parallel Search** � Execute queries via Exa API and extract full-text content
3. **Summarization** � Convert sources into cited evidence bullets
4. **Gap Analysis** � Determine if more research is needed (up to 2 rounds)
5. **Section Writing** � Draft sections with proper citations in parallel
6. **Assembly & Review** � Combine sections, renumber citations globally, review quality
7. **Revision** � Apply suggestions if quality checks fail

### Technology Stack

- **[DSPy](https://github.com/stanfordnlp/dspy)**: Framework for programming language models with signatures
- **[LangGraph](https://github.com/langchain-ai/langgraph)**: Orchestration for multi-agent workflows with parallelism
- **[Exa API](https://exa.ai/)**: Neural search engine for web research
- **Google Gemini**: LLM backend (2.5 Flash for research, 2.5 Pro for writing)
- **GEPA**: Efficient prompt optimization using reflection

## Requirements

- Python 3.10+
- API Keys:
  - Google Gemini API key
  - Exa API key (get from [dashboard.exa.ai](https://dashboard.exa.ai))

## Installation

```bash
# Clone the repository
cd dspy-gepa-researcher

# Install dependencies using uv (recommended)
uv add install dspy langgraph exa-py python-dateutil pydantic

# Or with pip
pip install dspy langgraph exa-py python-dateutil pydantic
```

## Configuration

Set the required environment variables:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export EXA_API_KEY="your-exa-api-key"

# Optional configuration
export GEMINI_WRITER_MODEL="gemini/gemini-flash-latest"      # Default
export GEMINI_RESEARCH_MODEL="gemini/gemini-flash-latest"    # Default
export RR_MAX_ROUNDS="2"                                      # Research rounds
export RR_SEARCH_K="6"                                        # Results per query
export RR_MAX_CHARS="12000"                                   # Max chars per source
```

## Usage

### Basic Usage

```python
import asyncio
from dspy_gepa_researcher import run_pipeline, SectionSpec

# Define your research sections
sections = [
    SectionSpec(
        name="Executive Summary",
        instructions="In 180-250 words, summarize the most decision-relevant takeaways."
    ),
    SectionSpec(
        name="Market Landscape",
        instructions="Define the space; 2023-2025 trends; include 4+ specific figures with sources."
    ),
    SectionSpec(
        name="Key Players & Differentiation",
        instructions="Compare 5-7 players; list 1-2 distinctive capabilities each."
    ),
]

# Run the pipeline
topic = "State of Edge AI Acceleration (2024-2025)"
result = asyncio.run(run_pipeline(
    topic=topic,
    sections=sections,
    optimization=False  # Set to True to enable GEPA optimization
))

# Report is saved to ./report.md
```

### Running the Example

```bash
uv run dspy_gepa_researcher.py
```

This will generate a report on "State of Edge AI Acceleration (2024-2025)" with 5 sections.

## How It Works

### 1. Query Generation
The `QUERY_GEN` agent creates 4-8 diverse search queries per section using:
- Quoted phrases for exact matches
- `site:` operators to target specific domains
- `intitle:` to find relevant titles
- Date ranges for time-specific research

### 2. Parallel Search & Summarization
Each query is executed in parallel:
- Exa API retrieves top-k documents with full text
- `SUMMARIZER` agent extracts key facts with `[S#]` citations
- Results are aggregated per section

### 3. Gap Analysis
The `GAP_ANALYZER` agent reviews evidence for each section:
- Determines if coverage is sufficient
- Generates followup queries if needed
- Proceeds to writing after max rounds or sufficient coverage

### 4. Section Writing
Sections are written in parallel:
- `WRITE_SECTION` drafts markdown with `[n]` citations
- `CITE_FIXER` ensures proper citation format
- Character count and citation stats are tracked

### 5. Assembly & Review
- Citations are renumbered globally across all sections
- `REVIEWER` agent checks for quality issues
- `REVISER` agent applies suggestions if needed
- Final report is saved with References section

## GEPA Optimization

The pipeline supports automatic prompt optimization using GEPA (Generalized Efficient Prompt Adaptation). When enabled, it optimizes prompts for all 7 agents:

1. `QUERY_GEN` - Search query generation
2. `SUMMARIZER` - Source summarization
3. `WRITE_SECTION` - Section writing
4. `GAP_ANALYZER` - Coverage analysis
5. `CITE_FIXER` - Citation formatting
6. `REVIEWER` - Quality review
7. `REVISER` - Report revision

Each agent is trained on 2-3 high-quality examples specific to its task.

To enable optimization:
```python
result = asyncio.run(run_pipeline(topic, sections, optimization=True))
```

## Output

The pipeline generates:

1. **report.md**: Complete markdown report with:
   - All sections with proper headings
   - Inline numeric citations `[1]`, `[2]`, etc.
   - References section with full metadata
   - Source URLs, publication dates, and domains

2. **Console Logs**: Detailed progress tracking:
   ```
   [QUERY] Planning search queries for 5 sections...
   [SEARCH] 'Market Size': "market size TAM 2024"...
   [SEARCH] � Found 6 documents
   [SEARCH] � Extracted 4 evidence bullets
   [GAP] Analyzing research coverage (Round 1/2)...
   [WRITE] Drafting section: 'Market Size'
   [WRITE] � 2341 chars, 8 citations
   [REVIEW] � Pass: true, Issues: 0, Suggestions: 2
   ```

3. **Evaluation Metrics**:
   - Overall quality score
   - Citation count
   - Character count
   - Structure checks

## Project Structure

```
dspy-gepa-researcher/
  dspy_gepa_researcher.py    # Main pipeline implementation
  README.md             # This file
  report.md             # Generated output (after running)
  LICENSE               # MIT License
```

## Customization

### Adding New Sections

```python
sections.append(SectionSpec(
    name="Technology Stack",
    instructions="List 5+ technologies with adoption rates, version info, and benchmarks"
))
```

### Adjusting Research Depth

```python
# More queries per section (default: 6)
export RR_SEARCH_K="10"

# More research rounds (default: 2)
export RR_MAX_ROUNDS="3"
```

### Changing Models

```python
export GEMINI_WRITER_MODEL="gemini/gemini-2.5-pro-preview-03-25"
export GEMINI_RESEARCH_MODEL="gemini/gemini-2.5-flash-preview-03-25"
```

## Limitations

- Requires active API keys (Gemini and Exa)
- Token costs scale with number of sections and research rounds
- Gemini models have output limits (8,192 tokens max per response)
- Quality depends on Exa search result quality

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built with [DSPy](https://github.com/stanfordnlp/dspy) by Stanford NLP
- Powered by [Exa](https://exa.ai/) neural search
- Orchestrated with [LangGraph](https://github.com/langchain-ai/langgraph)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{dspy_examples,
  title={DSPy Multi-Agent Research Pipeline},
  author={Your Name},
  year={2025},
  url={https://github.com/raja-patnaik/dspy-examples}
}
```
