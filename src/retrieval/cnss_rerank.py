"""
Cross-encoder reranking for Corpus 2 (CNSS FAQ), over a fused
BM25+dense candidate pool. Reuses build_reranker() from rerank.py
since the cross-encoder itself isn't corpus-specific.
"""

import time
from src.retrieval.rerank import build_reranker
from src.retrieval.cnss_bm25 import entry_search_text
from src.retrieval.cnss_fusion import fused_search


def rerank(query, fused_results, entries_by_id, reranker, top_n=5):
    pairs = []
    meta = []
    for rrf_rank, (entry_id, rrf_score) in enumerate(fused_results, start=1):
        entry = entries_by_id[entry_id]
        pairs.append([query, entry_search_text(entry)])
        meta.append({
            "id": entry_id,
            "question": entry["question"],
            "answer": entry["answer"],
            "category": entry["category"],
            "rrf_score": rrf_score,
            "rrf_rank": rrf_rank,
        })

    scores = reranker.compute_score(pairs, normalize=True)
    if isinstance(scores, float):
        scores = [scores]

    for m, s in zip(meta, scores):
        m["rerank_score"] = s

    return sorted(meta, key=lambda x: x["rerank_score"], reverse=True)[:top_n]


def retrieve(query, bm25_index, bm25_entries, dense_collection, dense_model,
             entries_by_id, reranker, pool=20, rrf_k=15, top_n=5):
    fused = fused_search(query, bm25_index, bm25_entries, dense_collection,
                          dense_model, k=rrf_k, pool=pool)
    return rerank(query, fused, entries_by_id, reranker, top_n=top_n)


if __name__ == "__main__":
    from src.retrieval.cnss_bm25 import load_entries as load_bm25_entries, build_bm25_index
    from src.retrieval.cnss_dense import (
        load_entries as load_dense_entries,
        embed_entries,
        build_index as build_dense_index,
    )
    from FlagEmbedding import BGEM3FlagModel

    bm25_entries = load_bm25_entries()
    bm25_index = build_bm25_index(bm25_entries)
    entries_by_id = {e["id"]: e for e in bm25_entries}

    dense_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    dense_entries = load_dense_entries()
    dense_vecs = embed_entries(dense_entries, dense_model)
    dense_collection = build_dense_index(dense_entries, dense_vecs)

    reranker = build_reranker()

    smoke_queries = [
        "pension de vieillesse délai",
        "prêt logement remboursement",
        "prêt voiture conjoint",
    ]

    for q in smoke_queries:
        print(f"\n=== Query: {q} ===")
        t0 = time.time()
        fused = fused_search(q, bm25_index, bm25_entries, dense_collection, dense_model, k=15, pool=20)
        t1 = time.time()
        reranked = rerank(q, fused, entries_by_id, reranker, top_n=5)
        t2 = time.time()

        print(f"  [retrieval: {(t1-t0)*1000:.0f}ms]  [rerank: {(t2-t1)*1000:.0f}ms]  [pairs scored: {len(fused)}]")
        for r in reranked:
            print(f"  {r['id']}  rerank={r['rerank_score']:.4f}  (was rrf_rank={r['rrf_rank']})  {r['question'][:70]}")