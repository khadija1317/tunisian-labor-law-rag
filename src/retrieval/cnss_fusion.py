from src.retrieval.cnss_bm25 import load_entries as load_bm25_entries, build_bm25_index, search as bm25_search
from src.retrieval.cnss_dense import (
    load_entries as load_dense_entries,
    embed_entries,
    build_index as build_dense_index,
    search as dense_search,
)
from FlagEmbedding import BGEM3FlagModel

RRF_K = 60


def rrf_fuse(ranked_lists, k=RRF_K):
    scores = {}
    for ranked_ids in ranked_lists:
        for rank, entry_id in enumerate(ranked_ids, start=1):
            scores[entry_id] = scores.get(entry_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def fused_search(query, bm25_index, bm25_entries, dense_collection, dense_model, k=15, pool=20):
    bm25_hits = bm25_search(query, bm25_index, bm25_entries, k=pool)
    bm25_ids = [h["id"] for h in bm25_hits]

    dense_hits = dense_search(query, dense_collection, dense_model, k=pool)
    dense_ids = [h["id"] for h in dense_hits]

    fused = rrf_fuse([bm25_ids, dense_ids])
    return fused[:k]


if __name__ == "__main__":
    bm25_entries = load_bm25_entries()
    bm25_index = build_bm25_index(bm25_entries)

    dense_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    dense_entries = load_dense_entries()
    dense_vecs = embed_entries(dense_entries, dense_model)
    dense_collection = build_dense_index(dense_entries, dense_vecs)

    smoke_queries = [
        "pension de vieillesse délai",
        "prêt logement remboursement",
        "prêt voiture conjoint",
    ]

    for q in smoke_queries:
        print(f"\n=== Query: {q} ===")
        results = fused_search(q, bm25_index, bm25_entries, dense_collection, dense_model)
        for entry_id, score in results:
            print(f"  {entry_id}  (rrf_score={score:.5f})")