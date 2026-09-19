"""
src/eval/score_latency.py

Per-stage latency measurement: router -> retrieval -> synthesizer -> verifier.
Calls each stage function directly (not handle_query()) so each stage's
wall-clock time can be measured in isolation.

Scope: bucket A + A2 only (14 pairs) -- these are the only pairs that
traverse all four stages. Bucket B stops after verifier (no citations),
bucket C stops after router (out_of_scope) -- timing those would only
measure router/retrieval, already covered here.
"""

import json
import time
import statistics

from src.agents import setup
from src.agents.router import route_query
from src.agents.synthesizer import synthesize
from src.agents.verifier import verify_answer
from src.agents.errors import LLMCallError

from src.retrieval.rerank import retrieve as retrieve_labor_code
from src.retrieval.cnss_rerank import retrieve as retrieve_cnss


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

BUCKET_A_AND_A2 = {
    "g001", "g002", "g003", "g004", "g005", "g006", "g007",
    "g009", "g010", "g011", "g012", "g013", "g020", "g021",
}


# ─────────────────────────────────────────────────────────────────────────────
# Load gold pairs
# ─────────────────────────────────────────────────────────────────────────────

with open("data/eval/qa_pairs.json", encoding="utf-8") as f:
    gold_pairs = {p["id"]: p for p in json.load(f)}


# ─────────────────────────────────────────────────────────────────────────────
# Timing storage
# ─────────────────────────────────────────────────────────────────────────────

latencies = {
    "router": [],
    "retrieval": [],
    "synthesizer": [],
    "verifier": [],
}

skipped = []


# ─────────────────────────────────────────────────────────────────────────────
# Per-pair timed run
# ─────────────────────────────────────────────────────────────────────────────

for pid in sorted(BUCKET_A_AND_A2):
    pair = gold_pairs[pid]
    query = pair["query"]

    print(f"\n{'='*60}\n{pid}: {query}\n{'='*60}")

    # --- router ---
    try:
        t0 = time.perf_counter()
        label = route_query(query)
        t1 = time.perf_counter()
        latencies["router"].append(t1 - t0)
        print(f"  router: {t1 - t0:.2f}s -> {label}")
    except LLMCallError as e:
        print(f"  SKIP {pid}: router failed -- {e}")
        skipped.append((pid, "router", str(e)))
        continue

    # --- retrieval ---
    t0 = time.perf_counter()
    if label == "labor_code":
        results = retrieve_labor_code(
            query, setup.labor_bm25_index, setup.labor_bm25_articles,
            setup.labor_dense_collection, setup.dense_model,
            setup.labor_articles_by_id, setup.reranker,
        )
    else:  # cnss
        results = retrieve_cnss(
            query, setup.cnss_bm25_index, setup.cnss_bm25_entries,
            setup.cnss_dense_collection, setup.dense_model,
            setup.cnss_entries_by_id, setup.reranker,
        )
    t1 = time.perf_counter()
    latencies["retrieval"].append(t1 - t0)
    print(f"  retrieval: {t1 - t0:.2f}s -> {len(results)} hits")

    # --- synthesizer ---
    try:
        t0 = time.perf_counter()
        synth_output = synthesize(query, results, corpus_type=label)
        t1 = time.perf_counter()
        latencies["synthesizer"].append(t1 - t0)
        print(f"  synthesizer: {t1 - t0:.2f}s")
    except LLMCallError as e:
        print(f"  SKIP {pid}: synthesizer failed -- {e}")
        skipped.append((pid, "synthesizer", str(e)))
        continue

    # --- verifier ---
    try:
        t0 = time.perf_counter()
        verdict = verify_answer(query, synth_output, results, corpus_type=label)
        t1 = time.perf_counter()
        latencies["verifier"].append(t1 - t0)
        print(f"  verifier: {t1 - t0:.2f}s -> grounded={verdict.grounded}")
    except LLMCallError as e:
        print(f"  SKIP {pid}: verifier failed -- {e}")
        skipped.append((pid, "verifier", str(e)))
        continue


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}\nSUMMARY (n={len(latencies['router'])} pairs completed all stages)\n{'='*60}")

for stage, times in latencies.items():
    if not times:
        print(f"  {stage}: no data")
        continue
    print(
        f"  {stage:12s}  mean={statistics.mean(times):.2f}s  "
        f"median={statistics.median(times):.2f}s  "
        f"min={min(times):.2f}s  max={max(times):.2f}s  n={len(times)}"
    )

total_mean = sum(statistics.mean(t) for t in latencies.values() if t)
print(f"\n  approx full-chain mean total: {total_mean:.2f}s")

if skipped:
    print(f"\n  {len(skipped)} pair(s) skipped due to LLMCallError:")
    for pid, stage, err in skipped:
        print(f"    {pid} ({stage}): {err}")