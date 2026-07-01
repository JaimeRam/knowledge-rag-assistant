import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM = (
    "You are an objective evaluator for RAG systems. "
    "Return only valid JSON, no other text."
)

_JUDGE_TEMPLATE = """Evaluate this RAG response:

Question: {query}

Retrieved context:
{context}

Generated answer: {answer}

Score 0.0-1.0 for each dimension:
- "faithfulness": is every claim in the answer grounded in the context? \
(1.0 = fully supported, 0.0 = unsupported)
- "relevance": does the answer directly address the question? \
(1.0 = perfectly relevant, 0.0 = off-topic)
- "context_relevance": do the retrieved context chunks actually help answer the question? \
(1.0 = all chunks are useful, 0.0 = chunks are generic or off-topic)

Respond ONLY with JSON: {{"faithfulness": <float>, "relevance": <float>, "context_relevance": <float>}}"""


class LLMJudge:
    """LLM-as-judge evaluator that scores RAG responses for faithfulness and relevance."""

    async def evaluate(
        self, query: str, context_texts: list, answer: str
    ) -> dict:
        """Return {"faithfulness": float, "relevance": float}, or {} on failure."""
        from app.rag.llm_client import LLMClient

        llm_client = LLMClient()
        try:
            context = "\n---\n".join(t for t in context_texts[:3] if t)
            prompt = _JUDGE_TEMPLATE.format(
                query=query,
                context=context[:2000],
                answer=answer[:1000],
            )
            response = await llm_client.chat(
                prompt, max_tokens=80, system=_JUDGE_SYSTEM
            )
            raw = response["content"].strip()
            logger.debug(f"[judge] raw LLM output: {raw}")
            # Extract JSON even when Claude wraps it in explanation text
            match = re.search(r"\{[^{}]+\}", raw)
            if not match:
                raise ValueError(f"No JSON object found in judge output: {raw!r}")
            scores = json.loads(match.group())
            return {
                "faithfulness": max(0.0, min(1.0, float(scores.get("faithfulness", 0.5)))),
                "relevance": max(0.0, min(1.0, float(scores.get("relevance", 0.5)))),
                "context_relevance": max(0.0, min(1.0, float(scores.get("context_relevance", 0.5)))),
            }
        except Exception as e:
            logger.warning(f"LLM judge evaluation failed: {e}")
            return {}
        finally:
            await llm_client.close()
