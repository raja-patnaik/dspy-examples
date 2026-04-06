# adversarial_debate.py
# Multi-Agent Adversarial Debate System with DSPy + GEPA
#
# Three agents — PRO advocate, CON advocate, and Judge — debate a topic
# over two rounds. GEPA optimizes all three simultaneously, creating
# adversarial pressure toward a Nash equilibrium.
#
# Requires: uv add dspy
# Env: export OPENAI_API_KEY=sk-...
# Run: uv run adversarial_debate.py

import os, json, re, statistics
from typing import Any

import dspy

# ------------------------------------------------------------------
# 0) Configure LMs
# ------------------------------------------------------------------

PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "openai/gpt-5.4-nano")
REFLECTION_MODEL = os.getenv("REFLECTION_MODEL", "openai/gpt-5.4-mini")

# GPT-5 family requires temperature=1.0 and max_tokens >= 16000
task_lm = dspy.LM(PRIMARY_MODEL, temperature=1.0, max_tokens=16000)
reflect_lm = dspy.LM(REFLECTION_MODEL, temperature=1.0, max_tokens=32000)
dspy.configure(lm=task_lm)

# ------------------------------------------------------------------
# 1) Signatures — declarative contracts for each debate role
# ------------------------------------------------------------------

class OpeningArgument(dspy.Signature):
    """Construct a compelling opening argument for the assigned position on the given topic.
    Structure with a clear thesis, 2-3 supporting points with evidence or reasoning,
    and a concluding summary. Do not strawman the opposing side."""

    topic: str = dspy.InputField(desc="The debate topic phrased as a proposition")
    position: str = dspy.InputField(desc="'PRO' (argue in favor) or 'CON' (argue against)")
    argument: str = dspy.OutputField(
        desc="Structured opening argument with thesis, supporting points, and conclusion. 150-300 words."
    )


class Rebuttal(dspy.Signature):
    """Respond to the opponent's argument with a targeted rebuttal.
    Address their specific claims, identify weaknesses, and reinforce your position
    with new evidence or reasoning not used in your opening."""

    topic: str = dspy.InputField(desc="The debate topic phrased as a proposition")
    position: str = dspy.InputField(desc="'PRO' or 'CON' — your assigned side")
    your_opening: str = dspy.InputField(desc="Your opening argument from the previous round")
    opponent_argument: str = dspy.InputField(desc="The opponent's most recent argument to rebut")
    rebuttal: str = dspy.OutputField(
        desc="Targeted rebuttal addressing opponent's points and reinforcing your own. 120-250 words."
    )


class JudgeRoundEval(dspy.Signature):
    """Evaluate one round of a debate between PRO and CON advocates.
    Score each side on argument quality, logical soundness, and evidence usage.
    Be impartial — do not favor either side based on the topic itself."""

    topic: str = dspy.InputField(desc="The debate topic")
    round_label: str = dspy.InputField(desc="'opening' or 'rebuttal'")
    pro_argument: str = dspy.InputField(desc="The PRO advocate's argument this round")
    con_argument: str = dspy.InputField(desc="The CON advocate's argument this round")
    round_assessment: str = dspy.OutputField(
        desc='JSON: {"pro_score": float 0-10, "con_score": float 0-10, "pro_notes": str, "con_notes": str}'
    )


class JudgeVerdict(dspy.Signature):
    """Deliver a final verdict on the full debate. Evaluate across all rounds.
    Score on argument quality, logical soundness, evidence usage, and rebuttal engagement.
    Declare a winner or tie. Remain impartial to the topic's substance."""

    topic: str = dspy.InputField(desc="The debate topic")
    debate_transcript: str = dspy.InputField(desc="Full transcript of all rounds")
    verdict: str = dspy.OutputField(
        desc='JSON: {"pro_total": float 0-10, "con_total": float 0-10, '
        '"winner": "PRO"|"CON"|"TIE", "reasoning": str}'
    )


# ------------------------------------------------------------------
# 2) Modules — composable debate agents
# ------------------------------------------------------------------

class DebateAdvocate(dspy.Module):
    """A debate advocate that can deliver openings and rebuttals."""

    def __init__(self):
        super().__init__()
        self.open = dspy.ChainOfThought(OpeningArgument)
        self.rebut = dspy.ChainOfThought(Rebuttal)


class DebateJudge(dspy.Module):
    """A judge that evaluates rounds and delivers final verdicts."""

    def __init__(self):
        super().__init__()
        self.eval_round = dspy.ChainOfThought(JudgeRoundEval)
        self.final_verdict = dspy.ChainOfThought(JudgeVerdict)


class DebateOrchestrator(dspy.Module):
    """Runs a full 2-round debate between PRO and CON advocates with a judge."""

    def __init__(self):
        super().__init__()
        self.pro = DebateAdvocate()
        self.con = DebateAdvocate()
        self.judge = DebateJudge()

    def forward(self, topic: str):
        # Round 1: Opening arguments
        pro_open = self.pro.open(topic=topic, position="PRO")
        con_open = self.con.open(topic=topic, position="CON")
        r1_eval = self.judge.eval_round(
            topic=topic, round_label="opening",
            pro_argument=pro_open.argument, con_argument=con_open.argument,
        )

        # Round 2: Rebuttals
        pro_reb = self.pro.rebut(
            topic=topic, position="PRO",
            your_opening=pro_open.argument, opponent_argument=con_open.argument,
        )
        con_reb = self.con.rebut(
            topic=topic, position="CON",
            your_opening=con_open.argument, opponent_argument=pro_open.argument,
        )
        r2_eval = self.judge.eval_round(
            topic=topic, round_label="rebuttal",
            pro_argument=pro_reb.rebuttal, con_argument=con_reb.rebuttal,
        )

        # Assemble transcript for final verdict
        transcript = build_transcript(
            topic,
            pro_open.argument, con_open.argument, r1_eval.round_assessment,
            pro_reb.rebuttal, con_reb.rebuttal, r2_eval.round_assessment,
        )

        final = self.judge.final_verdict(topic=topic, debate_transcript=transcript)

        return dspy.Prediction(
            pro_opening=pro_open.argument,
            con_opening=con_open.argument,
            pro_rebuttal=pro_reb.rebuttal,
            con_rebuttal=con_reb.rebuttal,
            round1_eval=r1_eval.round_assessment,
            round2_eval=r2_eval.round_assessment,
            verdict=final.verdict,
            transcript=transcript,
        )


# ------------------------------------------------------------------
# 3) Utilities
# ------------------------------------------------------------------

def build_transcript(
    topic: str,
    pro_opening: str, con_opening: str, r1_eval: str,
    pro_rebuttal: str, con_rebuttal: str, r2_eval: str,
) -> str:
    return (
        f"== DEBATE: {topic} ==\n\n"
        f"--- ROUND 1: OPENING ARGUMENTS ---\n"
        f"[PRO]: {pro_opening}\n\n"
        f"[CON]: {con_opening}\n\n"
        f"[JUDGE R1]: {r1_eval}\n\n"
        f"--- ROUND 2: REBUTTALS ---\n"
        f"[PRO REBUTTAL]: {pro_rebuttal}\n\n"
        f"[CON REBUTTAL]: {con_rebuttal}\n\n"
        f"[JUDGE R2]: {r2_eval}"
    )


def _extract_json(s: str) -> Any:
    """Extract first JSON object from a string; tolerant to surrounding text."""
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}


# ------------------------------------------------------------------
# 4) Fallacy detection patterns (heuristic, regex-based)
# ------------------------------------------------------------------

FALLACY_PATTERNS = {
    "ad_hominem": re.compile(
        r"\b(stupid|ignorant|foolish|naive|incompetent|idiotic)\b", re.I
    ),
    "strawman": re.compile(
        r"\b(clearly (they|the other side|opponents?) (think|believe|want|claim))\b", re.I
    ),
    "false_dilemma": re.compile(
        r"\b(either .{5,60} or .{5,60}|only two (options|choices|paths))\b", re.I
    ),
    "appeal_to_emotion": re.compile(
        r"\b(think of the children|imagine the horror|catastrophic|devastating consequences)\b", re.I
    ),
    "slippery_slope": re.compile(
        r"\b(inevitably lead|slippery slope|next thing you know|before long)\b", re.I
    ),
    "circular_reasoning": re.compile(
        r"\b(because it (is|just is)|by definition|obviously true|self-evident)\b", re.I
    ),
}

STOPWORDS = {
    "about", "above", "after", "again", "against", "along", "among", "around",
    "because", "before", "being", "below", "between", "could", "doing", "during",
    "every", "first", "would", "should", "their", "there", "these", "those",
    "through", "under", "until", "where", "which", "while", "might", "other",
    "still", "since", "often", "never", "always", "another", "without", "within",
}


# ------------------------------------------------------------------
# 5) Metric scoring components
# ------------------------------------------------------------------

def score_argument_quality(text: str, role: str, min_words: int = 150, max_words: int = 300):
    """Score structure markers and word count conformance."""
    words = len((text or "").split())
    score = 0.0
    notes = []

    # Word count
    if min_words <= words <= max_words:
        score += 0.3
    elif words < min_words:
        notes.append(f"{role}: too short ({words} words, need {min_words}+). Add more supporting points.")
        score += 0.1
    else:
        notes.append(f"{role}: too long ({words} words, max {max_words}). Be more concise.")
        score += 0.2

    # Structure markers
    has_thesis = bool(re.search(
        r"\b(therefore|thus|argue|position|believe|contend|propose|maintain)\b", text or "", re.I
    ))
    has_evidence = bool(re.search(
        r"\b(because|evidence|research|studies|data|example|for instance|shows? that|according)\b",
        text or "", re.I,
    ))
    has_conclusion = bool(re.search(
        r"\b(in conclusion|ultimately|in summary|therefore|to summarize|in closing)\b", text or "", re.I
    ))

    score += 0.3 if has_thesis else 0.0
    score += 0.2 if has_evidence else 0.0
    score += 0.2 if has_conclusion else 0.0

    if not has_thesis:
        notes.append(f"{role}: state a clear thesis or position.")
    if not has_evidence:
        notes.append(f"{role}: support claims with evidence or reasoning.")
    if not has_conclusion:
        notes.append(f"{role}: add a concluding statement.")

    return score, notes


def score_logical_soundness(text: str, role: str):
    """Detect fallacy patterns; each detection applies a penalty."""
    score = 1.0
    notes = []
    for name, pat in FALLACY_PATTERNS.items():
        m = pat.search(text or "")
        if m:
            score -= 0.15
            notes.append(
                f"{role}: possible {name.replace('_', ' ')} near "
                f"'{m.group()[:40]}'. Use evidence instead."
            )
    return max(0.0, score), notes


def score_engagement(opponent_argument: str, rebuttal_text: str):
    """Measure how much the rebuttal addresses the opponent's specific claims."""
    opponent_terms = set(re.findall(r"\b[a-z]{5,}\b", (opponent_argument or "").lower()))
    opponent_terms -= STOPWORDS

    if not opponent_terms:
        return 1.0, "No specific claims to engage with."

    rebuttal_lower = (rebuttal_text or "").lower()
    addressed = sum(1 for t in opponent_terms if t in rebuttal_lower)
    ratio = addressed / len(opponent_terms)

    if ratio < 0.15:
        fb = "Rebuttal does not engage with opponent's specific points. Address their claims directly."
    elif ratio < 0.35:
        fb = "Rebuttal partially engages. Reference more of the opponent's specific arguments."
    else:
        fb = "Good engagement with opponent's arguments."

    return min(1.0, ratio / 0.4), fb


def score_judge_fairness(verdict_raw: str):
    """Check if judge scores are proportional and reasoning is balanced."""
    parsed = _extract_json(verdict_raw)
    if not isinstance(parsed, dict):
        return 0.1, "Return valid JSON with pro_total, con_total, winner, reasoning."

    pro_total = float(parsed.get("pro_total", 0))
    con_total = float(parsed.get("con_total", 0))
    reasoning = str(parsed.get("reasoning", ""))
    total = pro_total + con_total

    fb_parts = []

    # Score spread: penalize extreme gaps
    if total == 0:
        return 0.1, "Judge provided no scores."
    spread = abs(pro_total - con_total) / total
    spread_score = max(0.0, 1.0 - spread * 2)

    if spread > 0.4:
        fb_parts.append(f"Score gap is {spread:.0%}; ensure both sides' strengths are acknowledged.")

    # Reasoning balance: both sides mentioned
    mentions_pro = bool(re.search(r"\bpro\b", reasoning, re.I))
    mentions_con = bool(re.search(r"\bcon\b", reasoning, re.I))
    balance_score = 1.0 if (mentions_pro and mentions_con) else 0.5

    if not mentions_pro or not mentions_con:
        fb_parts.append("Reasoning should explicitly address both PRO and CON arguments.")

    # Required keys check
    required = {"pro_total", "con_total", "winner", "reasoning"}
    present = required & set(parsed.keys())
    completeness = len(present) / len(required)
    missing = required - present
    if missing:
        fb_parts.append(f"Missing JSON keys: {', '.join(sorted(missing))}.")

    fairness = 0.3 * completeness + 0.4 * spread_score + 0.3 * balance_score

    if not fb_parts:
        fb_parts.append("Good balanced evaluation.")

    return fairness, " ".join(fb_parts)


# ------------------------------------------------------------------
# 6) GEPA metric — component-level feedback via pred_name
# ------------------------------------------------------------------

def debate_metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    """Multi-objective metric for the adversarial debate system.

    - When pred_name is None: returns scalar float (for dspy.Evaluate).
    - When pred_name is set: returns dspy.Prediction(score, feedback)
      with targeted feedback for GEPA reflection.
    """
    # --- Compute all dimension scores from the full prediction ---
    pro_opening = getattr(pred, "pro_opening", "") or ""
    con_opening = getattr(pred, "con_opening", "") or ""
    pro_rebuttal = getattr(pred, "pro_rebuttal", "") or ""
    con_rebuttal = getattr(pred, "con_rebuttal", "") or ""
    verdict_raw = getattr(pred, "verdict", "") or ""

    # Argument quality for all 4 arguments
    aq_pro_o, aq_pro_o_n = score_argument_quality(pro_opening, "PRO-opening")
    aq_con_o, aq_con_o_n = score_argument_quality(con_opening, "CON-opening")
    aq_pro_r, aq_pro_r_n = score_argument_quality(pro_rebuttal, "PRO-rebuttal", 120, 250)
    aq_con_r, aq_con_r_n = score_argument_quality(con_rebuttal, "CON-rebuttal", 120, 250)
    aq_avg = (aq_pro_o + aq_con_o + aq_pro_r + aq_con_r) / 4

    # Logical soundness for all 4
    ls_pro_o, ls_pro_o_n = score_logical_soundness(pro_opening, "PRO-opening")
    ls_con_o, ls_con_o_n = score_logical_soundness(con_opening, "CON-opening")
    ls_pro_r, ls_pro_r_n = score_logical_soundness(pro_rebuttal, "PRO-rebuttal")
    ls_con_r, ls_con_r_n = score_logical_soundness(con_rebuttal, "CON-rebuttal")
    ls_avg = (ls_pro_o + ls_con_o + ls_pro_r + ls_con_r) / 4

    # Engagement: rebuttals vs. opponent openings
    eng_pro, eng_pro_fb = score_engagement(con_opening, pro_rebuttal)
    eng_con, eng_con_fb = score_engagement(pro_opening, con_rebuttal)
    eng_avg = (eng_pro + eng_con) / 2

    # Judge fairness
    fair, fair_fb = score_judge_fairness(verdict_raw)

    total_score = 0.25 * aq_avg + 0.30 * ls_avg + 0.20 * eng_avg + 0.25 * fair
    total_score = max(0.0, min(1.0, total_score))

    # --- pred_name dispatch for GEPA component-level feedback ---
    if pred_name is None:
        return float(total_score)

    # pred_name includes module path, e.g. "pro.open.predict", "con.rebut.predict",
    # "judge.eval_round.predict", "judge.final_verdict.predict"
    if "open" in pred_name and "pro" in pred_name:
        fb = "\n".join(aq_pro_o_n + ls_pro_o_n) or "PRO opening is solid."
        return dspy.Prediction(score=total_score, feedback=fb)

    elif "open" in pred_name and "con" in pred_name:
        fb = "\n".join(aq_con_o_n + ls_con_o_n) or "CON opening is solid."
        return dspy.Prediction(score=total_score, feedback=fb)

    elif "rebut" in pred_name and "pro" in pred_name:
        fb = "\n".join(aq_pro_r_n + ls_pro_r_n + [eng_pro_fb]) or "PRO rebuttal is strong."
        return dspy.Prediction(score=total_score, feedback=fb)

    elif "rebut" in pred_name and "con" in pred_name:
        fb = "\n".join(aq_con_r_n + ls_con_r_n + [eng_con_fb]) or "CON rebuttal is strong."
        return dspy.Prediction(score=total_score, feedback=fb)

    elif "eval_round" in pred_name:
        fb = "Ensure round assessment is valid JSON with pro_score, con_score, pro_notes, con_notes."
        return dspy.Prediction(score=total_score, feedback=fb)

    elif "final_verdict" in pred_name:
        fb = fair_fb
        return dspy.Prediction(score=total_score, feedback=fb)

    # Fallback for any unexpected pred_name
    return dspy.Prediction(score=total_score, feedback=f"score={total_score:.3f}")


# ------------------------------------------------------------------
# 7) Training data — debate topics (no gold outputs, zero-shot)
# ------------------------------------------------------------------

train_set = [
    dspy.Example(
        topic="Space exploration should prioritize robotic missions over crewed missions"
    ).with_inputs("topic"),
    dspy.Example(
        topic="Universities should replace traditional lectures with project-based learning"
    ).with_inputs("topic"),
    dspy.Example(
        topic="Cities should prioritize public transit investment over road expansion"
    ).with_inputs("topic"),
]

val_set = [
    dspy.Example(
        topic="Open-source software is more reliable than proprietary software for critical infrastructure"
    ).with_inputs("topic"),
    dspy.Example(
        topic="Remote work produces better outcomes than in-office work for knowledge workers"
    ).with_inputs("topic"),
]


# ------------------------------------------------------------------
# 8) Pretty printing
# ------------------------------------------------------------------

def pretty_print_debate(pred):
    """Print a formatted debate transcript with scores."""
    print("\n" + "=" * 60)
    print("PRO OPENING:")
    print(pred.pro_opening[:500])
    print("\nCON OPENING:")
    print(pred.con_opening[:500])
    print("\nROUND 1 EVAL:", pred.round1_eval[:200])
    print("\nPRO REBUTTAL:")
    print(pred.pro_rebuttal[:500])
    print("\nCON REBUTTAL:")
    print(pred.con_rebuttal[:500])
    print("\nROUND 2 EVAL:", pred.round2_eval[:200])
    print("\nFINAL VERDICT:", pred.verdict[:300])
    print("=" * 60)


# ------------------------------------------------------------------
# 9) Main — baseline → GEPA optimize → compare
# ------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Multi-Agent Adversarial Debate with GEPA Optimization")
    print("=" * 60)

    program = DebateOrchestrator()

    # Baseline evaluation
    print("\n--- Baseline Evaluation ---")
    evaluate = dspy.Evaluate(
        devset=val_set,
        metric=debate_metric,
        num_threads=8,
        display_progress=True,
    )
    baseline_result = evaluate(program)
    print(f"Baseline score: {baseline_result}")

    # Sample baseline debate
    print("\n--- Sample Baseline Debate ---")
    sample = program(topic=val_set[0].topic)
    pretty_print_debate(sample)

    # GEPA optimization
    print("\n--- Optimizing with GEPA ---")
    optimizer = dspy.GEPA(
        metric=debate_metric,
        # auto="light",
        max_full_evals=12,
        reflection_lm=reflect_lm,
        num_threads=8,
        track_stats=True,
        use_merge=False,
    )

    optimized = optimizer.compile(
        student=DebateOrchestrator(),
        trainset=train_set,
        valset=val_set,
    )

    # Post-optimization evaluation
    print("\n--- Optimized Evaluation ---")
    optimized_result = evaluate(optimized)
    print(f"Optimized score: {optimized_result}")

    # Delta
    print(f"\nΔ score: {optimized_result.score - baseline_result.score:+.3f}")

    # Sample optimized debate
    print("\n--- Sample Optimized Debate ---")
    sample_opt = optimized(topic=val_set[0].topic)
    pretty_print_debate(sample_opt)


if __name__ == "__main__":
    main()
