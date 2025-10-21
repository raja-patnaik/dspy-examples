import os
import re
import sqlite3
import inspect
from typing import List, Tuple, Optional

import dspy

# -----------------------------------------------------------------------------
# 0) Model configuration
# -----------------------------------------------------------------------------
STUDENT_MODEL = os.getenv("DSPY_STUDENT_MODEL", "openai/gpt-4o-mini")
REFLECT_MODEL = os.getenv("DSPY_REFLECT_MODEL", "openai/gpt-4o")

student_lm = dspy.LM(STUDENT_MODEL, temperature=0.2, max_tokens=800)
reflection_lm = dspy.LM(REFLECT_MODEL, temperature=0.8, max_tokens=2000)
dspy.configure(lm=student_lm)

# GEPA import (newer/older DSPy layouts)
try:
    from dspy import GEPA
except Exception:
    try:
        from dspy.teleprompt import GEPA  # older path
    except Exception:
        GEPA = None


# -----------------------------------------------------------------------------
# 1) Tiny SQLite database and schema description
# -----------------------------------------------------------------------------
def setup_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    c = conn.cursor()

    c.executescript("""
    CREATE TABLE authors (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        country TEXT NOT NULL
    );
    CREATE TABLE books (
        id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        year INTEGER NOT NULL,
        author_id INTEGER NOT NULL,
        genre TEXT NOT NULL,
        pages INTEGER NOT NULL,
        price REAL NOT NULL,
        FOREIGN KEY(author_id) REFERENCES authors(id)
    );
    CREATE TABLE sales (
        book_id INTEGER NOT NULL,
        year INTEGER NOT NULL,
        sold INTEGER NOT NULL,
        FOREIGN KEY(book_id) REFERENCES books(id)
    );
    """)

    authors = [
        (1, "Margaret Atwood", "Canada"),
        (2, "Haruki Murakami", "Japan"),
        (3, "Chimamanda Ngozi Adichie", "Nigeria"),
        (4, "Neil Gaiman", "UK"),
        (5, "Alice Munro", "Canada"),
    ]
    c.executemany("INSERT INTO authors VALUES (?, ?, ?)", authors)

    books = [
        (1, "The Handmaid's Tale", 1985, 1, "Dystopia", 311, 9.99),
        (2, "Kafka on the Shore", 2002, 2, "Magical Realism", 505, 14.99),
        (3, "American Gods", 2001, 4, "Fantasy", 465, 12.99),
        (4, "Half of a Yellow Sun", 2006, 3, "Historical", 448, 13.99),
        (5, "The Testaments", 2019, 1, "Dystopia", 419, 15.99),
        (6, "Norwegian Wood", 1987, 2, "Romance", 296, 10.99),
        (7, "Dear Life", 2012, 5, "Short Stories", 336, 11.99),
        (8, "Neverwhere", 1996, 4, "Fantasy", 370, 9.49),
        (9, "Oryx and Crake", 2003, 1, "Dystopia", 389, 11.49),
    ]
    c.executemany("INSERT INTO books VALUES (?, ?, ?, ?, ?, ?, ?)", books)

    sales = [
        (1, 2024, 12000),
        (2, 2024, 15000),
        (3, 2024, 16000),
        (4, 2024, 11000),
        (5, 2024, 9000),
        (6, 2024, 13000),
        (7, 2024, 7000),
        (8, 2024, 8000),
        (9, 2024, 10000),
    ]
    c.executemany("INSERT INTO sales VALUES (?, ?, ?)", sales)

    conn.commit()
    return conn


def describe_schema(conn: sqlite3.Connection, sample_rows: int = 2) -> str:
    """Create a compact, LM-friendly schema string with a couple of sample rows per table."""
    c = conn.cursor()
    parts = []
    for table in ["authors", "books", "sales"]:
        c.execute(f"PRAGMA table_info({table})")
        cols = [f"{row[1]}:{row[2]}" for row in c.fetchall()]
        parts.append(f"TABLE {table}({', '.join(cols)})")
        c.execute(f"SELECT * FROM {table} LIMIT {sample_rows}")
        rows = c.fetchall()
        parts.append(f"EXAMPLE_ROWS {table}: {rows}")
    return "\n".join(parts)


# -----------------------------------------------------------------------------
# 2) DSPy program (Signature + Module)
# -----------------------------------------------------------------------------
class NL2SQL(dspy.Signature):
    """Generate a single safe SQLite SELECT to answer the question from the given schema."""

    schema = dspy.InputField(desc="SQLite schema and 1–2 sample rows per table")
    question = dspy.InputField(desc="Natural-language question about the data")
    sql = dspy.OutputField(
        desc=(
            "Only ONE statement. Start with SELECT or WITH. "
            "Use exact column/table names. "
            "Return only the SQL (no comments)."
        )
    )


class NL2SQLProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(
            NL2SQL
        )  # hidden reasoning; SQL only as output

    def forward(self, schema: str, question: str):
        return self.generate(schema=schema, question=question)


# -----------------------------------------------------------------------------
# 3) Questions + (optional) gold SQL
# -----------------------------------------------------------------------------
def questions_and_gold_sql() -> List[Tuple[str, Optional[str]]]:
    base: List[Tuple[str, Optional[str]]] = [
        (
            "List the titles of books written by authors from Canada, alphabetically.",
            "SELECT b.title FROM books b JOIN authors a ON a.id=b.author_id "
            "WHERE a.country='Canada' ORDER BY b.title;",
        ),
        (
            "Which author sold the most copies in 2024?",
            "SELECT a.name FROM authors a "
            "JOIN books b ON a.id=b.author_id "
            "JOIN sales s ON s.book_id=b.id "
            "WHERE s.year=2024 "
            "GROUP BY a.name ORDER BY SUM(s.sold) DESC LIMIT 1;",
        ),
        (
            "How many books per genre? Return genre and count, count descending.",
            "SELECT genre, COUNT(*) AS n FROM books GROUP BY genre ORDER BY n DESC;",
        ),
        (
            "What are the top 2 longest books by page count? Return title and pages.",
            "SELECT title, pages FROM books ORDER BY pages DESC LIMIT 2;",
        ),
        (
            "Average price of books published in or after 2010. Return a single number.",
            "SELECT ROUND(AVG(price), 2) AS avg_price FROM books WHERE year >= 2010;",
        ),
        (
            "List distinct countries represented by authors, alphabetically.",
            "SELECT DISTINCT country FROM authors ORDER BY country;",
        ),
        (
            "For Haruki Murakami, what is the average pages of his books? Return name and avg pages.",
            "SELECT a.name, ROUND(AVG(b.pages), 1) AS avg_pages "
            "FROM authors a JOIN books b ON a.id=b.author_id "
            "WHERE a.name='Haruki Murakami' GROUP BY a.name;",
        ),
        (
            "Find the cheapest Fantasy book (title + price).",
            "SELECT title, price FROM books WHERE genre='Fantasy' ORDER BY price ASC LIMIT 1;",
        ),
        (
            "Return titles that contain the word 'the' (case-insensitive), alphabetically.",
            "SELECT title FROM books WHERE LOWER(title) LIKE '%the%' ORDER BY title;",
        ),
        (
            "How many books did Margaret Atwood publish after 2000?",
            "SELECT COUNT(*) AS n FROM books b "
            "JOIN authors a ON a.id=b.author_id "
            "WHERE a.name='Margaret Atwood' AND year > 2000;",
        ),
    ]

    # Harder variations to create headroom (ordering, aliasing, shape, top-k)
    extras: List[Tuple[str, Optional[str]]] = [
        (
            "Return the top 3 authors by total copies sold in 2024 (name + total_sold), descending.",
            "SELECT a.name, SUM(s.sold) AS total_sold "
            "FROM authors a "
            "JOIN books b ON b.author_id = a.id "
            "JOIN sales s ON s.book_id = b.id "
            "WHERE s.year = 2024 "
            "GROUP BY a.name "
            "ORDER BY total_sold DESC "
            "LIMIT 3;",
        ),
        (
            "List titles containing the word 'the' (case-insensitive), alphabetically, return them as column lower_title.",
            "SELECT LOWER(title) AS lower_title "
            "FROM books "
            "WHERE LOWER(title) LIKE '%the%' "
            "ORDER BY lower_title ASC;",
        ),
        (
            "For each country, return country and number of authors as n_authors; sort by n_authors desc then country asc.",
            "SELECT country, COUNT(*) AS n_authors "
            "FROM authors "
            "GROUP BY country "
            "ORDER BY n_authors DESC, country ASC;",
        ),
        (
            "What percent of books are Dystopia? Return a single number named pct_dystopia (1 decimal place).",
            "SELECT ROUND(100.0 * SUM(CASE WHEN genre='Dystopia' THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_dystopia "
            "FROM books;",
        ),
    ]

    return base + extras


def run_sql(conn: sqlite3.Connection, sql: str):
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description] if cur.description else []
    return cols, rows


def build_examples(conn: sqlite3.Connection, schema_text: str):
    """Build DSPy examples and precompute gold results when available."""
    examples = []
    for q, gold_sql in questions_and_gold_sql():
        expected = None
        if gold_sql:
            cols, rows = run_sql(conn, gold_sql)
            ordered = "ORDER BY" in gold_sql.upper()
            expected = {"columns": cols, "rows": rows, "ordered": ordered}
        ex = dspy.Example(
            schema=schema_text, question=q, expected=expected
        ).with_inputs("schema", "question")
        examples.append(ex)
    return examples


# -----------------------------------------------------------------------------
# 4) Metric: safety + execution + strict(er) correctness + heuristic penalties
# -----------------------------------------------------------------------------
FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|PRAGMA|ATTACH|DETACH|CREATE|REPLACE|VACUUM|TRIGGER|INDEX|VIEW)\b",
    flags=re.IGNORECASE,
)


def _clean_sql(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```(?:sql)?", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"```$", "", t).strip()
    # Keep only the first statement; terminate neatly if a semicolon was present
    if ";" in t:
        t = t.split(";")[0].strip() + ";"
    return t


def sql_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
):
    """
    GEPA-friendly metric: returns {'score': float, 'feedback': str}.

    Scoring:
      +0.4  single safe SELECT/WITH, no forbidden tokens
      +0.3  executes without error
      +0.3  exact result match (rows+columns); if not exact but sets equal → +0.15
    Heuristic penalties (applied after success): -0.05 each for missing ORDER BY when asked,
    wrong direction, missing LIMIT k when asked, missing DISTINCT when asked, wrong output shape for "single number".
    """
    sql_raw = getattr(pred, "sql", "") or ""
    sql = _clean_sql(sql_raw)
    question = getattr(gold, "question", "")
    ql = question.lower()

    score = 0.0
    fb = []

    # 1) Safety / format
    if not (sql.lower().startswith("select") or sql.lower().startswith("with")):
        fb.append("SQL must start with SELECT or WITH.")
    elif FORBIDDEN.search(sql):
        fb.append("Forbidden tokens present (DDL/DML/PRAGMA/etc.).")
    else:
        score += 0.4

    # 2) Execution
    exec_cols, exec_rows, exec_err = [], [], None
    if score >= 0.4:
        try:
            conn = (
                setup_db()
            )  # fresh DB prevents accidental writes (even though we forbid DDL/DML)
            cur = conn.cursor()
            cur.execute(sql)
            exec_rows = cur.fetchall()
            exec_cols = [d[0] for d in cur.description] if cur.description else []
            score += 0.3
        except Exception as e:
            exec_err = str(e)
            fb.append(f"Execution error: {exec_err}")

    # 2.1) Heuristic penalties open headroom for GEPA
    penalty = 0.0
    if exec_err is None:
        su = sql.upper()

        asks_order = any(
            w in ql
            for w in [
                "alphabet",
                "ascending",
                "descending",
                "top",
                "highest",
                "lowest",
                "most",
            ]
        )
        if asks_order and "ORDER BY" not in su:
            penalty += 0.05
            fb.append("Question asks for ordering; add an ORDER BY.")

        if any(w in ql for w in ["descending", "highest", "top", "most", "largest"]):
            if "ORDER BY" in su and "DESC" not in su:
                penalty += 0.05
                fb.append("Use ORDER BY ... DESC for descending/top/most queries.")

        if "alphabet" in ql or "ascending" in ql:
            if "ORDER BY" in su and "DESC" in su:
                penalty += 0.05
                fb.append("Use ORDER BY ... ASC for alphabetical/ascending queries.")

        m = re.search(r"\btop\s+(\d+)\b", ql)
        if m:
            k = int(m.group(1))
            if f"LIMIT {k}" not in su:
                penalty += 0.05
                fb.append(f"Add LIMIT {k} as requested (top {k}).")

        if "distinct" in ql and "DISTINCT" not in su:
            penalty += 0.05
            fb.append("Add DISTINCT as requested.")

        if "single number" in ql:
            if not (len(exec_rows) == 1 and len(exec_cols) == 1):
                penalty += 0.05
                fb.append(
                    "Return exactly one column and one row for 'single number' requests."
                )

    # 3) Correctness vs. gold (if provided)
    s_correct = 0.0
    gold_expected = getattr(gold, "expected", None)
    if exec_err is None and gold_expected:
        gold_rows = gold_expected["rows"]
        gold_cols = gold_expected["columns"]
        require_order = bool(gold_expected.get("ordered", False))

        same_cols = exec_cols == gold_cols

        if require_order:
            same_rows = exec_rows == gold_rows
        else:
            same_rows = sorted(map(tuple, exec_rows)) == sorted(map(tuple, gold_rows))

        if same_cols and same_rows:
            s_correct = 0.3
        else:
            # partial credit: set-equality on rows and columns
            set_rows_equal = sorted(map(tuple, exec_rows)) == sorted(
                map(tuple, gold_rows)
            )
            set_cols_equal = set(exec_cols) == set(gold_cols)
            if set_rows_equal and set_cols_equal:
                s_correct = 0.15
                if not same_cols:
                    fb.append(
                        "Column order/aliases differ; add explicit aliases to match expected columns."
                    )
                if require_order and not same_rows:
                    fb.append("Row order differs; add the correct ORDER BY.")
            else:
                fb.append(
                    f"Result mismatch. Expected (sample): {gold_rows[:3]} | Got (sample): {exec_rows[:3]}"
                )

    elif exec_err is None and not gold_expected:
        # Unlabeled partial credit to keep signal flowing
        if len(exec_rows) > 0:
            s_correct = 0.15
            fb.append(
                "No gold available; granting partial credit for non-empty result."
            )
        else:
            fb.append("Query returned 0 rows; consider joins/filters.")

    score += s_correct
    score = max(0.0, min(1.0, score - penalty))  # apply penalties, clamp to [0, 1]

    if score == 1.0:
        fb.append("Perfect score. Keep current strategy.")
    elif score < 0.4:
        fb.append("Rewrite as ONE safe SELECT/WITH; avoid DDL/DML/PRAGMA.")

    return {"score": float(score), "feedback": "\n".join(fb)}


def sql_metric_scalar(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Numeric-only wrapper compatible with GEPA's 5-arg metric signature."""
    return float(
        sql_metric(
            gold,
            pred,
            trace=trace,
            pred_name=pred_name,
            pred_trace=pred_trace,
        )["score"]
    )


def sql_metric_dual(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Dual-mode metric for older GEPA builds:
        - Evaluation path (no trace): return float score
        - Reflection path (has trace/pred_name/pred_trace): return {'score', 'feedback'}
    """
    res = sql_metric(
        gold, pred, trace=trace, pred_name=pred_name, pred_trace=pred_trace
    )
    # If called by Evaluate (no reflection context), return float only:
    if (trace is None) and (pred_name is None) and (pred_trace is None):
        return float(res["score"])
    # If called by GEPA reflection, return rich feedback:
    return res


# -----------------------------------------------------------------------------
# 5) Main — build data, baseline eval, GEPA optimization, post eval
# -----------------------------------------------------------------------------
def main():
    if GEPA is None:
        raise RuntimeError(
            "Could not import GEPA from DSPy. Please upgrade/install a DSPy version that includes GEPA."
        )

    # Build schema and examples
    conn = setup_db()
    schema_text = describe_schema(conn)
    all_examples = build_examples(conn, schema_text)

    # Train/dev split (tweak to taste)
    trainset = all_examples[:8]
    devset = all_examples[8:]

    # Baseline program
    program = NL2SQLProgram()

    # Baseline detailed feedback on devset
    print("== Baseline on devset ==")
    baseline_scores = []
    for ex in devset:
        pred = program(schema=ex.schema, question=ex.question)
        res = sql_metric(ex, pred)
        baseline_scores.append(res["score"])
        print("\nQ:", ex.question)
        print("SQL:\n", _clean_sql(getattr(pred, "sql", "")))
        print(f"Score: {res['score']:.3f}")
        print("Feedback:\n" + res["feedback"])
    print(f"\nBaseline mean score: {sum(baseline_scores) / len(baseline_scores):.3f}")

    # GEPA optimizer (reflective prompt evolution)
    # --- Compatibility wiring: metric must be numeric; feedback wired if supported.
    gepa_kwargs = dict(
        metric=sql_metric_scalar,  # <-- numeric metric (prevents the dict summation crash)
        auto="medium",
        reflection_lm=reflection_lm,
        track_stats=True,
        add_format_failure_as_feedback=True,
    )

    init_sig = inspect.signature(GEPA.__init__).parameters
    if "feedback_metric" in init_sig:
        gepa_kwargs["feedback_metric"] = sql_metric
    elif "feedback_producer" in init_sig:  # some older builds
        gepa_kwargs["feedback_producer"] = sql_metric
    else:
        print(
            "⚠️  GEPA build does not expose 'feedback_metric'/'feedback_producer'. "
            "Optimization will use the scalar metric only (still works, less rich guidance)."
        )

    gepa = GEPA(**gepa_kwargs)

    optimized_program = gepa.compile(program, trainset=trainset, valset=devset)

    # Post‑GEPA detailed feedback on devset
    print("\n== Post‑GEPA on devset ==")
    post_scores = []
    for ex in devset:
        pred = optimized_program(schema=ex.schema, question=ex.question)
        res = sql_metric(ex, pred)
        post_scores.append(res["score"])
        print("\nQ:", ex.question)
        print("SQL:\n", _clean_sql(getattr(pred, "sql", "")))
        print(f"Score: {res['score']:.3f}")
        print("Feedback:\n" + res["feedback"])
    print(f"\nPost‑GEPA mean score: {sum(post_scores) / len(post_scores):.3f}")

    # Quick before/after on a single sample
    sample = devset[0]
    before = program(schema=sample.schema, question=sample.question)
    after = optimized_program(schema=sample.schema, question=sample.question)
    print("\n== Before/After example ==")
    print("Q:", sample.question)
    print("Before SQL:\n", _clean_sql(getattr(before, "sql", "")))
    print("After  SQL:\n", _clean_sql(getattr(after, "sql", "")))


if __name__ == "__main__":
    main()
