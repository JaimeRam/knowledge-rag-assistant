"""RAG evaluation script using RAGAS.

Evaluates the Digimon RAG pipeline across 4 standard metrics:
  - faithfulness      : Is the answer grounded in the retrieved context?
  - answer_relevancy  : Is the answer relevant to the question?
  - context_precision : Are the retrieved chunks precise (not noisy)?
  - context_recall    : Do the chunks cover the ground-truth answer?

Usage (main venv):
    python scripts/evaluate_rag.py            # build dataset + evaluate (~3 min + ~55 min)
    python scripts/evaluate_rag.py --eval-only  # reuse cached dataset, skip rebuild

Requirements: make services + make ingest (Qdrant must have data).
"""

import sys
import warnings

# Suppress ragas deprecation noise before any ragas import
warnings.filterwarnings("ignore", category=DeprecationWarning)

from types import ModuleType

# Stub out VertexAI before ragas imports it — ragas hard-imports ChatVertexAI
# at module level but we never use it. This prevents the Google Cloud chain.
for _mod, _attrs in [
    ("langchain_community.chat_models.vertexai", {"ChatVertexAI": object}),
    ("langchain_community.llms.vertexai", {"VertexAI": object}),
    ("langchain_community.llms", {"VertexAI": object}),
]:
    if _mod not in sys.modules:
        _stub = ModuleType(_mod)
        for _attr, _val in _attrs.items():
            setattr(_stub, _attr, type(_attr, (_val,), {}))
        sys.modules[_mod] = _stub

import argparse
import asyncio
import json
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_anthropic import ChatAnthropic
from langchain_voyageai import VoyageAIEmbeddings

from app.core.config import get_settings
from app.rag.retriever import Retriever
from app.rag.prompt_builder import PromptBuilder
from app.rag.llm_client import LLMClient
from scripts.eval_dataset import EVAL_DATASET

_DEFAULT_CACHE = os.path.join(os.path.dirname(__file__), ".eval_cache.json")


def _save_cache(samples: list, path: str) -> None:
    data = [
        {
            "user_input": s.user_input,
            "response": s.response,
            "retrieved_contexts": s.retrieved_contexts,
            "reference": s.reference,
        }
        for s in samples
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Dataset cached to {path}")


def _load_cache(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    return [
        SingleTurnSample(
            user_input=d["user_input"],
            response=d["response"],
            retrieved_contexts=d["retrieved_contexts"],
            reference=d["reference"],
        )
        for d in data
    ]


async def build_samples() -> list:
    """Run the RAG pipeline for each golden question and cache results."""
    retriever = Retriever()
    llm_client = LLMClient()
    samples = []

    print(f"Building evaluation dataset from {len(EVAL_DATASET)} questions...")

    for i, item in enumerate(EVAL_DATASET, 1):
        print(f"  [{i}/{len(EVAL_DATASET)}] {item['question'][:60]}...")
        if i > 1:
            time.sleep(21)  # Voyage AI free tier: 3 RPM → 20s between calls

        chunks = await retriever.retrieve(item["question"], limit=5)
        contexts = [
            c.get("payload", {}).get("chunk_text", "")
            for c in chunks
            if c.get("payload", {}).get("chunk_text")
        ]

        prompt = PromptBuilder.build_chat_prompt(item["question"], chunks)
        response = await llm_client.chat(prompt)
        answer = response.get("content", "")

        samples.append(SingleTurnSample(
            user_input=item["question"],
            response=answer,
            retrieved_contexts=contexts,
            reference=item["ground_truth"],
        ))

    await llm_client.close()
    await retriever.close()
    return samples


def run_evaluation(samples: list) -> None:
    settings = get_settings()

    dataset = EvaluationDataset(samples=samples)

    llm_judge = LangchainLLMWrapper(ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=settings.anthropic_api_key,
    ))
    embeddings_judge = LangchainEmbeddingsWrapper(VoyageAIEmbeddings(
        voyage_api_key=settings.voyage_api_key,
        model="voyage-3",
    ))

    metrics = [
        Faithfulness(llm=llm_judge),
        ResponseRelevancy(llm=llm_judge, embeddings=embeddings_judge),
        LLMContextPrecisionWithReference(llm=llm_judge),
        LLMContextRecall(llm=llm_judge),
    ]

    # max_workers=1: Voyage AI free tier is 3 RPM — sequential prevents timeouts
    run_config = RunConfig(timeout=180, max_retries=2, max_wait=30, max_workers=1)

    print("\nRunning RAGAS evaluation (sequential mode — Voyage AI 3 RPM limit)...")
    result = evaluate(dataset=dataset, metrics=metrics, run_config=run_config)

    df = result.to_pandas()
    metric_cols = df.select_dtypes(include="number").columns.tolist()

    print("\n=== Per-question scores ===")
    print(df[["user_input"] + metric_cols].to_string(index=False))

    print("\n=== Overall averages ===")
    for col in metric_cols:
        avg = df[col].mean()
        status = "✓" if avg >= 0.7 else "⚠"
        print(f"  {status}  {col:<40} {avg:.3f}")

    print("\nDone. Faithfulness < 0.7 → review prompt. Context recall < 0.6 → review ingest/embeddings.")


async def main(eval_only: bool, cache_path: str) -> None:
    if eval_only:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cache found at {cache_path}. Run without --eval-only first."
            )
        print(f"Loading dataset from cache: {cache_path}")
        samples = _load_cache(cache_path)
    else:
        samples = await build_samples()
        _save_cache(samples, cache_path)

    run_evaluation(samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Digimon RAG pipeline with RAGAS.")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip dataset rebuild; load from cache (must have run once without this flag).",
    )
    parser.add_argument(
        "--cache",
        default=_DEFAULT_CACHE,
        metavar="FILE",
        help=f"Cache file path (default: {_DEFAULT_CACHE})",
    )
    args = parser.parse_args()

    asyncio.run(main(eval_only=args.eval_only, cache_path=args.cache))
