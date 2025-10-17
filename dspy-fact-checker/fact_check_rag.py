import os
import dspy
import wikipedia
from dspy.dsp.utils import dotdict


# --------------------------------------------------------------------------------------
# Wikipedia API Retriever that returns objects with a `.long_text` field (DSPy expects this)
# --------------------------------------------------------------------------------------
class WikipediaRetriever:
    """Simple retriever using the Wikipedia Python API."""

    def __init__(self, max_chars_per_passage: int = 1500, language: str = "en"):
        self.max_chars = max_chars_per_passage
        wikipedia.set_lang(language)

    def __call__(self, query: str, k: int = 8, **kwargs):
        """Return a list[dotdict] where each item has a `.long_text` attribute."""
        try:
            titles = wikipedia.search(query, results=k) or []
        except Exception:
            titles = []

        passages = []

        for title in titles[:k]:
            page = None
            picked_title = title

            try:
                page = wikipedia.page(title, auto_suggest=False)
            except wikipedia.exceptions.DisambiguationError as e:
                # Resolve disambiguation by trying the first few options
                for opt in e.options[:3]:
                    try:
                        page = wikipedia.page(opt, auto_suggest=False)
                        picked_title = opt
                        break
                    except Exception:
                        continue
            except Exception:
                pass

            if not page:
                continue

            text = (page.summary or "").strip()
            if not text:
                continue

            if len(text) > self.max_chars:
                text = text[: self.max_chars].rstrip() + "..."

            # IMPORTANT: Return a structure with `.long_text`
            # We also embed title + URL into the long_text so they survive DSPy's mapping.
            long_text = f"{picked_title}: {text} (Source: {page.url})"
            passages.append(
                dotdict(
                    {"long_text": long_text, "title": picked_title, "url": page.url}
                )
            )

            if len(passages) >= k:
                break

        return passages


# --------------------------------------------------------------------------------------
# OpenAI LM configuration (uses OPENAI_API_KEY from env)
# --------------------------------------------------------------------------------------
# You can also pass api_key=... directly to dspy.LM if you prefer
lm = dspy.LM("openai/gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"))
dspy.configure(
    lm=lm
)  # Official pattern for configuring the default LM. :contentReference[oaicite:1]{index=1}

# Configure the retriever for dspy.Retrieve
wiki_rm = WikipediaRetriever(max_chars_per_passage=1500, language="en")
dspy.settings.configure(rm=wiki_rm)


# --------------------------------------------------------------------------------------
# Signatures
# --------------------------------------------------------------------------------------
class GenerateAnswer(dspy.Signature):
    """Answer the question strictly from the provided context.
    - Use only facts present in the context.
    - If the context doesn't contain the answer, say you don't know.
    - Keep the answer concise (2-5 sentences).
    """

    context = dspy.InputField(desc="retrieved passages from Wikipedia")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="factual answer derived ONLY from the context")


class VerifyAnswer(dspy.Signature):
    """Given 'context' and an 'answer', list any claims in the answer that are NOT supported by the context.
    Output 'None' if every claim is supported.
    """

    context = dspy.InputField(desc="retrieved passages from Wikipedia")
    answer = dspy.InputField(desc="candidate answer to verify")
    unsupported_claims = dspy.OutputField(
        desc="List unsupported claims or 'None' if fully supported."
    )


# --------------------------------------------------------------------------------------
# Self-correcting RAG with dspy.Refine
# --------------------------------------------------------------------------------------
class FactCheckedRAG(dspy.Module):
    def __init__(self, k_passages: int = 4, max_attempts: int = 3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=k_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.verify_answer = dspy.ChainOfThought(VerifyAnswer)

        # Reward: 1.0 when verifier returns "None"
        def reward_fn(args, pred):
            context_text = args["context"]
            verification = self.verify_answer(context=context_text, answer=pred.answer)
            uc = (verification.unsupported_claims or "").strip().lower()
            return 1.0 if (uc == "" or uc == "none" or uc.startswith("none")) else 0.0

        # Retry generation with automatic feedback until reward meets threshold
        self.refine_generate = dspy.Refine(
            module=self.generate_answer,
            N=max_attempts,
            reward_fn=reward_fn,
            threshold=1.0,
        )

    def forward(self, question: str):
        # Retrieve evidence (DSPy maps dotdicts -> list[str] via `.long_text`)
        passages = self.retrieve(question).passages  # List[str]
        context_text = "\n\n".join([f"[{i + 1}] {p}" for i, p in enumerate(passages)])

        # Generate (and refine if needed) until verified
        pred = self.refine_generate(context=context_text, question=question)
        answer = pred.answer

        # Final verification for reporting
        final_check = self.verify_answer(context=context_text, answer=answer)

        return dspy.Prediction(
            answer=answer,
            context=context_text,
            unsupported_claims=final_check.unsupported_claims,
        )


# --------------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    program = FactCheckedRAG(k_passages=4, max_attempts=3)

    # Try any well-known topic
    question = (
        "When did Apollo 11 land on the Moon, and who were the astronauts involved?"
    )
    # question = "Who discovered penicillin and in which year was it first reported?"
    # question = "When was the first FIFA World Cup held, and where?"

    result = program(question)

    print(f"\nQuestion: {question}")
    print("-" * 80)
    print("Final Answer:\n", result.answer)
    print("-" * 80)
    print("Unsupported Claims:", result.unsupported_claims)
    print("-" * 80)
    print("Context used:\n", result.context)
