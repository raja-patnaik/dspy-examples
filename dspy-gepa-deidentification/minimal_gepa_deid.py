import re
import dspy


# 0) Pick task + reflection models (reflection ≈ stronger)
task_lm = dspy.LM("openai/gpt-4o-mini")
reflect_lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=task_lm)  # global default LM for modules :contentReference[oaicite:8]{index=8}

# 1) Signature: what the module does (not how to prompt)
class DeIDSignature(dspy.Signature):
    """Rewrite an incident report to remove PII while preserving causal structure and action items."""
    report = dspy.InputField(desc="Raw incident report text.")
    rules  = dspy.InputField(desc="Redaction rules and required output format.")
    clean_report = dspy.OutputField(
        desc="Redacted report using [EMAIL], [PHONE], [NAME]. Keep 'Root cause:' + 'Action items:' and bullets."
    )

# 2) Module: we’ll let GEPA evolve its internal instructions
class DeIDProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.rewriter = dspy.ChainOfThought(DeIDSignature)  # adds .reasoning field to the prediction :contentReference[oaicite:9]{index=9}
    def forward(self, report, rules):
        return self.rewriter(report=report, rules=rules)

student = DeIDProgram()

# 3) Tiny “dataset”: GEPA doesn’t require labels, just examples to evaluate on
RULES = """Redact emails, phone numbers, and full names. Use placeholders [EMAIL], [PHONE], [NAME].
Keep section headers and bullets. Output format:
Root cause: ...
Action items: ...
- bullets for action items"""

trainset = [
    dspy.Example(
        report="Root cause: Alice Chen emailed ops (alice.chen@acme.io).\nAction items:\n- Call +1 (415) 555-0199 to notify vendor.",
        rules=RULES
    ).with_inputs("report", "rules"),
    dspy.Example(
        report="Root cause: Misconfigured S3 bucket by Bob A.\nAction items:\n- Rotate keys\n- email secops@company.com with incident ID 12345",
        rules=RULES
    ).with_inputs("report", "rules"),
]

devset = [
    dspy.Example(
        report="Root cause: OT sensor alert phoned to 212-555-0101 by Carol Q.\nAction items:\n- File ticket\n- email ops@example.org",
        rules=RULES
    ).with_inputs("report", "rules"),
]
# Note: .with_inputs tells DSPy which fields are inputs for evaluation/compilation. :contentReference[oaicite:10]{index=10}

# 4) Metric with feedback: score + *text* guidance for GEPA
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?:\\+?\\d{1,3}[-. (]*)?\\d{3}[-. )]*\\d{3}[-. ]*\\d{4}")
NAME  = re.compile(r"\\b([A-Z][a-z]+ [A-Z][a-z]+)\\b")

def pii_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    text = (pred.clean_report or "").strip()
    leaks = []
    if EMAIL.search(text): 
        leaks.append("email")
    if PHONE.search(text): 
        leaks.append("phone")
    if NAME.search(gold.report) and "[NAME]" not in text:
        leaks.append("name")

    keeps_root = "Root cause:" in text
    keeps_actions = "Action items:" in text

    # Score ∈ [0,1]: 0.6 for zero leaks + 0.2 each for keeping the two sections
    score = (0.6 if not leaks else 0.0) + (0.2 if keeps_root else 0.0) + (0.2 if keeps_actions else 0.0)

    feedback = []
    if leaks:
        feedback.append(f"PII leaked: {', '.join(leaks)}. Replace PII with [EMAIL], [PHONE], [NAME].")
    if not keeps_root or not keeps_actions:
        missing = []
        if not keeps_root: 
            missing.append("keep 'Root cause:'")
        if not keeps_actions:
            missing.append("keep 'Action items:'")
        feedback.append("Also " + " and ".join(missing) + ".")
    if not feedback:
        feedback.append("Great: no PII and structure preserved. Prefer succinct edits; avoid adding facts.")

    return dspy.Prediction(score=score, feedback=" ".join(feedback))  # GEPA reads this feedback to evolve instructions.


# Slightly stricter composite metric
def composite_pii_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    text = (pred.clean_report or "").strip()
    issues = []

    # 1) PII leak checks (extend with better detectors as needed)
    leaks = []
    if EMAIL.search(text): 
        leaks.append("email")
    if PHONE.search(text): 
        leaks.append("phone")
    if NAME.search(gold.report) and "[NAME]" not in text: 
        leaks.append("name")
    if leaks: 
        issues.append(f"PII leaked: {', '.join(leaks)}; replace with placeholders.")

    # 2) Structure invariants
    if "Root cause:" not in text:  
        issues.append("Missing header: 'Root cause:'.")
    if "Action items:" not in text: 
        issues.append("Missing header: 'Action items:'.")

    # 3) Formatting: require bullets for action items
    if "Action items:" in text:
        after = text.split("Action items:", 1)[1]
        if "-" not in after and "\n•" not in after:
            issues.append("Action items must be bulleted with '-' or '•'.")

    # 4) No fabrication: forbid adding new emails/phones beyond placeholders
    hallucination = EMAIL.findall(text) or PHONE.findall(text)
    if hallucination: 
        issues.append("Do not introduce new PII; use placeholders only.")

    # Score scheme
    base = 1.0
    penalty = 0.25 * len(issues)  # tune per your tolerance
    score = max(0.0, base - penalty)
    feedback = " ".join(issues) if issues else (
        "Great: no leaks, headers intact, bullets present; keep edits minimal and factual."
    )
    return dspy.Prediction(score=score, feedback=feedback)


# 5) Run GEPA (reflection model must be provided)
gepa = dspy.GEPA(
    # metric=pii_metric,
    metric=composite_pii_metric,
    auto="light",
    reflection_lm=reflect_lm,
    track_stats=True,
    track_best_outputs=True  # also useful as an inference-time search to surface best candidates per input
)  # See GEPA API for params like candidate_selection_strategy='pareto'. :contentReference[oaicite:12]{index=12}

optimized = gepa.compile(student, trainset=trainset, valset=devset)

# 6) Try it
test_report = (
    "Root cause: Dave Miller called 650-555-0000 to report breach.\n"
    "Action items:\n- email dave@contoso.com\n- notify legal"
)
print(optimized(report=test_report, rules=RULES).clean_report)

# Optional: Inspect the Pareto/best outputs per instance
# print(optimized.detailed_results.best_outputs_valset)  # requires track_best_outputs=True :contentReference[oaicite:13]{index=13}