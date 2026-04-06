# Multi-Agent Adversarial Debate with GEPA

A DSPy example that demonstrates **GEPA optimizing adversarial multi-agent dynamics**.

Three agents debate a topic over two rounds:
- **PRO Advocate** — argues in favor of the proposition
- **CON Advocate** — argues against
- **Judge** — evaluates each round and delivers a final verdict

GEPA optimizes all three agents simultaneously. Improving one advocate creates pressure on the other, while the judge must stay calibrated — producing a prompt-level Nash equilibrium.

## What makes this novel

Most prompt optimizers work on single-agent tasks. Here, GEPA faces a **multi-agent game**: the metric rewards *both* sides being strong (not just one winning), and the judge being fair. This creates adversarial co-adaptation pressure that's unique to this architecture.

## Architecture

```
Round 1: Opening Arguments
  PRO → opening(topic, "PRO")
  CON → opening(topic, "CON")
  Judge → evaluate_round("opening", pro_arg, con_arg)

Round 2: Rebuttals
  PRO → rebuttal(topic, "PRO", own_opening, opponent_opening)
  CON → rebuttal(topic, "CON", own_opening, opponent_opening)
  Judge → evaluate_round("rebuttal", pro_rebuttal, con_rebuttal)

Final: Judge → verdict(topic, full_transcript)
```

7 LM calls per debate, all optimizable by GEPA.

## Metric design

Four heuristic dimensions (no LLM-as-judge in the metric):

| Dimension | Weight | How it's measured |
|---|---|---|
| Argument Quality | 25% | Structure markers (thesis/evidence/conclusion) + word count |
| Logical Soundness | 30% | Regex-based fallacy detection (ad hominem, strawman, etc.) |
| Rebuttal Engagement | 20% | Term overlap between opponent's argument and the rebuttal |
| Judge Fairness | 25% | Score spread + balanced reasoning + JSON completeness |

GEPA receives **per-agent feedback** via `pred_name` dispatch — each predictor gets targeted guidance for what to improve.

## Setup

```bash
# Install dependencies
uv add dspy

# Set your API key
export OPENAI_API_KEY=sk-...

# Run
uv run adversarial_debate.py
```

## Configuration

Environment variables:
- `PRIMARY_MODEL` — task LM (default: `openai/gpt-5.4-nano`)
- `REFLECTION_MODEL` — GEPA reflection LM (default: `openai/gpt-5.4-mini`)
- `OPENAI_API_KEY` — your OpenAI API key

## GEPA settings

```python
optimizer = dspy.GEPA(
    metric=debate_metric,
    auto="light",
    reflection_lm=reflect_lm,
    num_threads=8,
    track_stats=True,
    use_merge=False,
)
```

- `auto="light"` keeps cost manageable for the demo
- `use_merge=False` simplifies the optimization loop
- All 3 agents are optimized in a single GEPA run

## Debate topics

**Training** (3 topics):
1. Space exploration: robotic vs. crewed missions
2. Universities: lectures vs. project-based learning
3. Cities: public transit vs. road expansion

**Validation** (2 topics):
4. Open-source vs. proprietary software for critical infrastructure
5. Remote work vs. in-office for knowledge workers

No gold outputs — GEPA learns entirely from metric feedback (zero-shot).
