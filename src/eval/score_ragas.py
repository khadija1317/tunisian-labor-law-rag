import json
import time

from langchain_core.embeddings import Embeddings
from groq import Groq

from ragas import evaluate, EvaluationDataset
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness
from ragas.metrics import (
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
)
from ragas.run_config import RunConfig

from src.agents import setup
from src.agents.dispatcher import handle_query


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "openai/gpt-oss-20b"

DEBUG = False

METRICS = [Faithfulness]
# METRICS = [Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference]


BUCKET_A_AND_A2 = {
    "g001", "g002", "g003", "g004", "g005", "g006", "g007",
    "g009", "g010", "g011", "g012", "g013", "g020", "g021",
}


# ─────────────────────────────────────────────────────────────────────────────
# Load gold evaluation questions
# ─────────────────────────────────────────────────────────────────────────────

with open("data/eval/qa_pairs.json", encoding="utf-8") as f:
    gold_pairs = {p["id"]: p for p in json.load(f)}


# ─────────────────────────────────────────────────────────────────────────────
# BGE-M3 adapter for RAGAS
# ─────────────────────────────────────────────────────────────────────────────

class BGEM3Embeddings(Embeddings):
    """Adapter allowing RAGAS to reuse the already-loaded BGE-M3 model."""

    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        out = self.model.encode(
            texts, return_dense=True, return_sparse=False, return_colbert_vecs=False,
        )
        return out["dense_vecs"].tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]


# ─────────────────────────────────────────────────────────────────────────────
# Format contexts already returned by handle_query() -- no re-retrieval
# ─────────────────────────────────────────────────────────────────────────────

def format_retrieved_texts(retrieved_results, label):
    if label == "labor_code":
        return [r["text"] for r in retrieved_results]
    else:
        return [f"Q: {r['question']}\nR: {r['answer']}" for r in retrieved_results]


# ─────────────────────────────────────────────────────────────────────────────
# Run the RAG system and build the RAGAS dataset (with progress printing)
# ─────────────────────────────────────────────────────────────────────────────

import os as _os
SAMPLES_CACHE = "data/eval/ragas_samples_cache.json"

if _os.path.exists(SAMPLES_CACHE):
    with open(SAMPLES_CACHE, encoding="utf-8") as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} cached samples from {SAMPLES_CACHE} -- skipping rebuild.")
else:
    samples = []
    total = len(BUCKET_A_AND_A2)
    start_time = time.time()

    for i, pid in enumerate(sorted(BUCKET_A_AND_A2), start=1):

        pair = gold_pairs[pid]
        pair_start = time.time()

        print(f"[{i}/{total}] {pid}: querying...", flush=True)
        result = handle_query(pair["query"])

        if result["status"] != "answered":
            print(
                f"  SKIP {pid}: expected answered, got {result['status']} -- excluding.",
                flush=True,
            )
            continue

        contexts = format_retrieved_texts(
            result["retrieved_results"],
            result["label"],
        )

        samples.append({
            "user_input": pair["query"],
            "retrieved_contexts": contexts,
            "response": result["final_answer"],
    })

        pair_elapsed = time.time() - pair_start
        total_elapsed = time.time() - start_time

        print(
        f"  done in {pair_elapsed:.1f}s  "
        f"(total elapsed: {total_elapsed/60:.1f} min)",
        flush=True,
    )

#build_elapsed = time.time() - start_time
#print(f"\nBuilt {len(samples)} samples in {build_elapsed/60:.1f} min total (bucket A + A2).")


import json as _json
SAMPLES_CACHE = "data/eval/ragas_samples_cache.json"
with open(SAMPLES_CACHE, "w", encoding="utf-8") as f:
    _json.dump(samples, f, ensure_ascii=False)
print(f"Cached samples to {SAMPLES_CACHE}")


# ─────────────────────────────────────────────────────────────────────────────
# RAGAS evaluator LLM + embeddings
# ─────────────────────────────────────────────────────────────────────────────

from langchain_groq import ChatGroq

evaluator_llm = LangchainLLMWrapper(
    ChatGroq(model=MODEL_NAME, temperature=0, max_tokens=16384)
)

evaluator_embeddings = LangchainEmbeddingsWrapper(
    BGEM3Embeddings(setup.dense_model)
)


# ─────────────────────────────────────────────────────────────────────────────
# RAGAS evaluation
# ─────────────────────────────────────────────────────────────────────────────

#run_config = RunConfig(max_workers=1 if DEBUG else 2)
run_config = RunConfig(
    max_workers=2,
    timeout=180,
)
eval_start = time.time()

result = evaluate(
    dataset=EvaluationDataset.from_list(samples),
    metrics=[m(llm=evaluator_llm) for m in METRICS],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
    run_config=run_config,
)

eval_elapsed = time.time() - eval_start
print(f"\nEvaluation took {eval_elapsed/60:.1f} min.")


# ─────────────────────────────────────────────────────────────────────────────
# Display + save results
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== RAGAS results (mean across samples) ===")
print(result)

df = result.to_pandas()
print("\n=== Per-pair scores ===")
print(df.to_string())

df.to_csv("data/eval/ragas_results.csv", index=False)
print("\nSaved per-pair scores to data/eval/ragas_results.csv")