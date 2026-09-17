"""
src/eval/score_retrieval.py

Retrieval-only scoring: Recall@5 and MRR@5 against expected_doc_ids,
for the 14 gold pairs where a single correct-document answer exists
(bucket A: single-answer, A2: known multi-source g013).
Zero Groq cost -- calls retrieve() directly, bypasses the full chain.
"""

import json
from src.agents import setup
from src.retrieval.rerank import retrieve as retrieve_labor_code
from src.retrieval.cnss_rerank import retrieve as retrieve_cnss

RETRIEVAL_ONLY_IDS = {
    "g001", "g002", "g003", "g004", "g005", "g006", "g007",
    "g009", "g010", "g011", "g012", "g013", "g020", "g021",
}

with open("data/eval/qa_pairs.json", encoding="utf-8") as f:
    gold_pairs = {p["id"]: p for p in json.load(f)}


def recall_and_rr(expected_ids, retrieved_ids):
    expected_set = set(expected_ids)
    if not expected_set:
        return None, None
    recall = len(set(retrieved_ids) & expected_set) / len(expected_set)
    rr = 0.0
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in expected_set:
            rr = 1.0 / rank
            break
    return recall, rr


results = []
for pid in sorted(RETRIEVAL_ONLY_IDS):
    pair = gold_pairs[pid]
    query = pair["query"]
    expected_ids = pair["expected_doc_ids"]

    if pair["expected_route_label"] == "labor_code":
        hits = retrieve_labor_code(
            query, setup.labor_bm25_index, setup.labor_bm25_articles,
            setup.labor_dense_collection, setup.dense_model,
            setup.labor_articles_by_id, setup.reranker,
        )
        retrieved_ids = [h["article_id"] for h in hits]
    else:
        hits = retrieve_cnss(
            query, setup.cnss_bm25_index, setup.cnss_bm25_entries,
            setup.cnss_dense_collection, setup.dense_model,
            setup.cnss_entries_by_id, setup.reranker,
        )
        retrieved_ids = [h["id"] for h in hits]

    recall, rr = recall_and_rr(expected_ids, retrieved_ids)
    results.append({"id": pid, "recall": recall, "rr": rr,
                     "expected": expected_ids, "retrieved": retrieved_ids})
    print(f"{pid}: recall={recall:.2f}  rr={rr:.2f}  "
          f"expected={expected_ids}  retrieved={retrieved_ids}")

mean_recall = sum(r["recall"] for r in results) / len(results)
mean_rr = sum(r["rr"] for r in results) / len(results)
print(f"\nMean Recall@5: {mean_recall:.3f}")
print(f"Mean MRR@5:    {mean_rr:.3f}   (n={len(results)})")