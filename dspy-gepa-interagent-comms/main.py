# research_agents_gepa.p
# A minimal multi-agent research assistant with DSPy + GEPA optimizing
# inter-agent communication protocols (routing + request + update).
#
# pip install -U dspy-ai gepa
#
# Env: export OPENAI_API_KEY=sk-...
# Optional: set PRIMARY_MODEL / REFLECTION_MODEL (defaults below).

import os, json, math, re, random, statistics
from typing import List, Dict, Any, Tuple, NamedTuple

import dspy

# ---------------------------
# 1) Configure LMs
# ---------------------------
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "openai/gpt-4o-mini")
REFLECTION_MODEL = os.getenv("REFLECTION_MODEL", "openai/gpt-4o-mini")
PRIMARY_TEMPERATURE = float(os.getenv("PRIMARY_TEMPERATURE", "0.3"))
REFLECTION_TEMPERATURE = float(os.getenv("REFLECTION_TEMPERATURE", "0.7"))

dspy.configure(lm=dspy.LM(PRIMARY_MODEL, temperature=PRIMARY_TEMPERATURE, max_tokens=1500))
reflection_lm = dspy.LM(REFLECTION_MODEL, temperature=REFLECTION_TEMPERATURE, max_tokens=4000)

# ---------------------------
# 2) Tiny Knowledge Base + BM25-like retriever (pure Python)
# ---------------------------
KB_DOCS: List[Dict[str, str]] = [
    {"id": "lg101", "title": "LangGraph at a glance", "text": "LangGraph is a framework for building stateful multi-actor LLM systems represented as graphs of nodes (agents) and edges. It supports cycles, persistent state, streaming, and human-in-the-loop interrupts."},
    {"id": "dspy101", "title": "DSPy Signatures & Modules", "text": "In DSPy, a Signature defines the inputs and outputs of an LM call. A Module implements a strategy such as Predict or ChainOfThought and can be composed. Optimizers compile programs to improve instructions and examples."},
    {"id": "gepa101", "title": "GEPA basics", "text": "GEPA is a reflective prompt optimizer that evolves textual components via natural-language reflection and Pareto selection across multiple objectives such as accuracy and cost."},
    {"id": "rag101", "title": "RAG pipeline recipe", "text": "A simple research assistant pipeline includes: query generation, retrieval over a corpus (e.g., BM25), synthesis, and critique for factuality."},
    {"id": "bm25", "title": "BM25 summary", "text": "BM25 is a bag-of-words retrieval function that scores documents using term frequency, inverse document frequency, and document length normalization with parameters k1 and b."},
    {"id": "cite", "title": "Citations", "text": "Good research answers should include citations to sources, using identifiers like [lg101] and [gepa101]."},
    {"id": "math", "title": "Specialist tasks", "text": "Some tasks, like arithmetic, are better handled by a math specialist rather than retrieval."},
    {"id": "critic", "title": "Critic agent", "text": "A critic checks whether claims are supported by retrieved evidence and flags missing citations."},
    {"id": "jsonproto", "title": "Message formatting", "text": "JSON protocols improve inter-agent routing by enforcing fields 'target' and 'payload' with keys 'query' and 'k'."},
    {"id": "efficiency", "title": "Efficiency principle", "text": "Short, canonical messages reduce token cost without hurting success."},
]

def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", s.lower())

class MiniBM25:
    def __init__(self, docs: List[Dict[str, str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.N = len(docs)
        self.doc_tokens = []
        self.df = {}
        self.avgdl = 0.0
        self._build()

    def _build(self):
        lengths = []
        for d in self.docs:
            tokens = _tokenize(d["title"] + " " + d["text"])
            self.doc_tokens.append(tokens)
            lengths.append(len(tokens))
            for t in set(tokens):
                self.df[t] = self.df.get(t, 0) + 1
        self.avgdl = sum(lengths) / max(1, len(lengths))

    def _idf(self, t: str) -> float:
        # BM25 idf
        n = self.df.get(t, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def _score_doc(self, q_tokens: List[str], tokens: List[str]) -> float:
        score = 0.0
        dl = len(tokens)
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for t in set(q_tokens):
            if t not in tf:
                continue
            idf = self._idf(t)
            numerator = tf[t] * (self.k1 + 1)
            denom = tf[t] + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            score += idf * numerator / (denom or 1)
        return score

    def search(self, query: str, k: int = 3) -> List[Dict[str, str]]:
        q_tokens = _tokenize(query)
        scored = [(self._score_doc(q_tokens, tok), i) for i, tok in enumerate(self.doc_tokens)]
        scored.sort(reverse=True)
        out = []
        for s, i in scored[:k]:
            d = self.docs[i]
            out.append({"id": d["id"], "title": d["title"], "text": d["text"], "score": round(s, 3)})
        return out

KB = MiniBM25(KB_DOCS)

# ---------------------------
# 3) Communication Signatures (protocols we will optimize)
# ---------------------------

class RouteQuery(dspy.Signature):
    """Supervisor selects which specialist to call next and how to call them.

    Return STRICT JSON ONLY with keys:
      - "target": one of ["retriever", "math"]
      - "payload": object with at least {"query": <string>, "k": <int>} for retriever
                   or {"problem": <string>} for math.
    Keep the payload minimal and avoid explanations.
    """
    problem_brief: str = dspy.InputField()
    candidate_specialists: str = dspy.InputField(desc="A short list of available specialists and their capabilities.")
    routed_call: str = dspy.OutputField(desc="Strict JSON as specified. No prose.")

class AgentCommunicationProtocol(dspy.Signature):
    """Agent A requesting help from Agent B.

    Produce STRICT JSON ONLY:
      {"query": <string>, "k": <int>}
    The query must be concise, with only essential keywords derived from the task_context.
    """
    task_context: str = dspy.InputField()
    agent_b_capabilities: str = dspy.InputField()
    optimized_request: str = dspy.OutputField()

class ProvideUpdate(dspy.Signature):
    """Agent B reports back to Agent A.

    Return STRICT JSON ONLY:
      {"update": "<concise synthesis with citations like [docid]>", "next_action": "stop"|"refine"}
    The 'update' should draw only from the provided partial_result and cite with [docid] tags.
    """
    partial_result: str = dspy.InputField()
    blockers: str = dspy.InputField()
    update: str = dspy.OutputField()

class SolveMath(dspy.Signature):
    """Solve small arithmetic exactly.

    Return STRICT JSON ONLY: {"answer": <integer>}
    """
    problem: str = dspy.InputField()
    solution: str = dspy.OutputField()

# ---------------------------
# 4) Utility: JSON parsing tolerant to minor formatting issues
# ---------------------------

def _extract_json(s: str) -> Any:
    """Extract first {...} or [...] JSON block; fallback to literal dict keys parsing."""
    if not isinstance(s, str):
        return s
    # try direct JSON
    try:
        return json.loads(s)
    except Exception:
        pass
    # try to find a JSON block
    m = re.search(r"(\{.*\}|\[.*\])", s, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # very rough fallback for "key: value" pairs
    obj = {}
    for line in s.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            obj[k.strip().strip('"')] = v.strip().strip('", ')
    return obj

def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    # crude 4-char-per-token heuristic
    return math.ceil(len(text) / 4)

# ---------------------------
# 5) Specialists (retriever + math)
# ---------------------------

def run_retriever(request_json: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
    query = request_json.get("query") or ""
    k = int(request_json.get("k") or 3)
    k = max(1, min(k, 5))
    docs = KB.search(query, k=k)
    # Bundle contexts with [docid] tags for the summarizer
    context = "\n\n".join(f"[{d['id']}] {d['text']}" for d in docs)
    return docs, context

# ---------------------------
# 6) Research program (multi-agent orchestration)
# ---------------------------

class ResearchProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        # The *protocol* layers are learnable LM calls that GEPA will mutate.
        self.route_query = dspy.Predict(RouteQuery)
        self.request_help = dspy.Predict(AgentCommunicationProtocol)
        self.provide_update = dspy.Predict(ProvideUpdate)
        self.solve_math = dspy.Predict(SolveMath)

        # Human-readable manifest of specialists
        self.specialists_manifest = (
            "retriever: BM25 search over a small internal corpus; returns [docid] snippets.\n"
            "math: deterministic arithmetic for small integer problems."
        )

    def forward(self, question: str):
        messages: List[Dict[str, Any]] = []

        # 1) Supervisor routes
        routed = self.route_query(
            problem_brief=question,
            candidate_specialists=self.specialists_manifest
        ).routed_call
        messages.append({"role": "supervisor", "protocol": "RouteQuery", "content": routed})

        routed_obj = _extract_json(routed) or {}
        target = (routed_obj.get("target") or "").strip().lower()
        payload = routed_obj.get("payload") or {}

        # 2a) Math specialist branch
        if target == "math":
            # allow fallback if payload missing
            problem = payload.get("problem") or question
            math_out = self.solve_math(problem=problem).solution
            messages.append({"role": "math", "protocol": "SolveMath", "content": math_out})
            sol = _extract_json(math_out)
            final_answer = f"{sol.get('answer')}" if isinstance(sol, dict) and "answer" in sol else str(sol)
            transcript_json = json.dumps(messages)
            return dspy.Prediction(
                final_answer=final_answer,
                transcript_json=transcript_json,
                route_chosen="math"
            )

        # 2b) Retriever + synthesis branch (default)
        req = self.request_help(
            task_context=question,
            agent_b_capabilities="Return STRICT JSON {'query': str, 'k': int}."
        ).optimized_request
        messages.append({"role": "agentA", "protocol": "RequestHelp", "content": req})

        req_obj = _extract_json(req) or {}
        if "query" not in req_obj:
            # graceful fallback
            req_obj = {"query": question, "k": 3}

        docs, context = run_retriever(req_obj)
        messages.append({"role": "retriever", "protocol": "BM25", "content": str([d['id'] for d in docs])})

        upd = self.provide_update(partial_result=context, blockers="none").update
        messages.append({"role": "specialist", "protocol": "ProvideUpdate", "content": upd})

        upd_obj = _extract_json(upd)
        if isinstance(upd_obj, dict) and "update" in upd_obj:
            final_answer = upd_obj["update"]
        else:
            final_answer = str(upd)

        transcript_json = json.dumps(messages)
        return dspy.Prediction(
            final_answer=final_answer,
            transcript_json=transcript_json,
            route_chosen="retriever"
        )

# ---------------------------
# 7) Dataset (train/dev) — small but meaningful
# ---------------------------

def make_example(question: str, must_include: List[str], requires: str):
    # requires ∈ {"retrieval", "math"}
    return dspy.Example(
        question=question,
        must_include=must_include,
        requires=requires
    ).with_inputs("question")

DATASET: List[dspy.Example] = [
    make_example(
        "What is LangGraph and name two features it adds for multi-agent workflows?",
        ["graphs of nodes", "persistent state"], "retrieval"
    ),
    make_example(
        "Explain what a DSPy Signature is and how it relates to Modules.",
        ["Signature defines the inputs and outputs", "Module implements a strategy"], "retrieval"
    ),
    make_example(
        "Briefly define GEPA and how it selects candidates.",
        ["reflective prompt optimizer", "Pareto"], "retrieval"
    ),
    make_example(
        "Outline a simple research assistant pipeline.",
        ["query generation", "retrieval", "synthesis", "critique"], "retrieval"
    ),
    make_example(
        "What are the core components of BM25 and its parameters?",
        ["term frequency", "inverse document frequency", "k1", "b"], "retrieval"
    ),
    make_example(
        "Why include citations in research answers?",
        ["citations", "[lg101]"], "retrieval"
    ),
    make_example(
        "Compute 137 + 286.",
        ["423"], "math"
    ),
    make_example(
        "Which tasks benefit from a math specialist vs retrieval?",
        ["arithmetic"], "retrieval"
    ),
    make_example(
        "What does the critic agent do?",
        ["supported by retrieved evidence", "flags missing citations"], "retrieval"
    ),
    make_example(
        "Why use JSON protocols for routing?",
        ["target", "payload"], "retrieval"
    ),
]

random.Random(7).shuffle(DATASET)
SPLIT = int(0.6 * len(DATASET))
TRAINSET = DATASET[:SPLIT]
DEVSET = DATASET[SPLIT:]

# ---------------------------
# 8) Multi-objective GEPA metric
#     - success (1/0): all must_include appear in final_answer
#     - token cost: total tokens of all messages (approx)
#     - turns: number of messages
#     - routing accuracy: did it choose the right specialist?
# ---------------------------

ALPHA_SUCCESS = 1.00       # weight for success
BETA_TOKENS   = 0.0006     # penalty per token (approx chars/4)
GAMMA_TURNS   = 0.03       # penalty per message turn
DELTA_ROUTE   = 0.25       # bonus for correct routing

class FeedbackResult(NamedTuple):
    """Result structure for GEPA metric feedback.

    NamedTuple supports both dict-style (fb["score"]) and
    attribute-style (fb.score) access for compatibility.
    """
    score: float
    feedback: str

def _evaluate_pred(gold: dspy.Example, pred: dspy.Prediction) -> Dict[str, Any]:
    # success
    ans = (getattr(pred, "final_answer", "") or "").lower()
    must = [m.lower() for m in (getattr(gold, "must_include", []) or [])]
    success = float(all(m in ans for m in must))

    # transcript stats
    transcript_json = getattr(pred, "transcript_json", "[]")
    try:
        messages = json.loads(transcript_json)
    except Exception:
        messages = []
    tokens = sum(_approx_tokens(str(m.get("content", ""))) for m in messages)
    turns = len(messages)

    # routing accuracy
    requires = getattr(gold, "requires", "retrieval")
    chosen = (getattr(pred, "route_chosen", "retriever") or "").lower()
    route_acc = float((requires == "math" and chosen == "math") or (requires == "retrieval" and chosen == "retriever"))

    return dict(success=success, tokens=tokens, turns=turns, route_acc=route_acc)

def gepa_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any | None = None
):
    stats = _evaluate_pred(gold, pred)
    score = (
        ALPHA_SUCCESS * stats["success"]
        - BETA_TOKENS * stats["tokens"]
        - GAMMA_TURNS * stats["turns"]
        + DELTA_ROUTE  * stats["route_acc"]
    )

    # Provide targeted feedback at the *component* level (GEPA uses this).
    fb_lines = []
    if pred_name == "RouteQuery":
        if stats["route_acc"] < 1.0:
            fb_lines.append("Prefer the 'math' specialist for arithmetic-only problems; otherwise use 'retriever'.")
        fb_lines.append("Return strict JSON with keys {target, payload}. Keep payload minimal.")
    if pred_name == "AgentCommunicationProtocol":
        if stats["tokens"] > 600:
            fb_lines.append("Shorten the request JSON: only {'query': ..., 'k': ...}. No justifications.")
    if pred_name == "ProvideUpdate":
        fb_lines.append("Synthesis must cite with [docid] tags only from provided context. Keep it concise.")

    # GEPA expects {'score': float, 'feedback': str} at the predictor level.
    if pred_name is not None:
        return FeedbackResult(
            score=float(score),
            feedback="\n".join(fb_lines) or f"score={score:.3f}"
        )
    # For regular Evaluate (no component-level optimization), return scalar.
    return float(score)

# A plain scalar version for convenience (baselines, printing)
def scalar_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    return float(gepa_metric(gold, pred))

# ---------------------------
# 9) Run: Baseline → GEPA optimize → Evaluate
# ---------------------------

def run_once(program: dspy.Module, devset: List[dspy.Example]) -> Dict[str, Any]:
    scores = []
    details = []
    for ex in devset:
        out = program(question=ex.question)
        s = scalar_metric(ex, out)
        scores.append(s)
        details.append((ex, out, s))
    return {"avg": statistics.mean(scores) if scores else 0.0, "scores": scores, "details": details}

def pretty_print_example(ex: dspy.Example, pred: dspy.Prediction):
    print("\n--- Example ---")
    print("Q:", ex.question)
    print("Must include:", ex.must_include, "| Requires:", ex.requires)
    print("Final answer:", pred.final_answer)
    try:
        messages = json.loads(pred.transcript_json)
    except Exception:
        messages = []
    print("Transcript:")
    for i, m in enumerate(messages, 1):
        print(f"  {i:02d}. [{m.get('role')}/{m.get('protocol')}] {m.get('content')}")
    stats = _evaluate_pred(ex, pred)
    print("Stats:", stats)

def main():
    print("Setting up program...")
    base_prog = ResearchProgram()

    print("\nBaseline on DEV:")
    base_res = run_once(base_prog, DEVSET)
    print(f"Baseline average score: {base_res['avg']:.3f} over {len(DEVSET)} examples")

    # Optimize the communication protocols (the LM predictors inside ResearchProgram)
    print("\nOptimizing protocols with GEPA...")
    gepa = dspy.GEPA(
        metric=gepa_metric,
        reflection_lm=reflection_lm,
        candidate_selection_strategy="pareto",
        track_stats=True,
        auto="light",
        seed=0
    )
    # Use TRAINSET to reflect/learn, and evaluate candidates on DEVSET (validation).
    opt_prog = gepa.compile(student=ResearchProgram(), trainset=TRAINSET, valset=DEVSET)

    print("\nRe-evaluating on DEV with the optimized program:")
    opt_res = run_once(opt_prog, DEVSET)
    print(f"Optimized average score: {opt_res['avg']:.3f} over {len(DEVSET)} examples")
    delta = opt_res["avg"] - base_res["avg"]
    print(f"Δ score: {delta:+.3f}")

    # Show a couple of transcripts before/after
    print("\nSample baseline transcript:")
    ex0 = DEVSET[0]
    pretty_print_example(ex0, base_prog(question=ex0.question))

    print("\nSample optimized transcript:")
    pretty_print_example(ex0, opt_prog(question=ex0.question))

if __name__ == "__main__":
    main()
