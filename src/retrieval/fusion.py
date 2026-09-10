"""
src/retrieval/fusion.py

Reciprocal Rank Fusion (RRF) combining BM25 (lexical) and dense (bge-m3)
rankings for Corpus 1. Fuses by rank position, not raw score, to avoid
the BM25-vs-cosine scale mismatch (see Day 2 resource notes).
"""

from src.retrieval.bm25 import load_articles as load_bm25_articles, build_bm25_index, search as bm25_search
from src.retrieval.dense import (
    load_articles as load_dense_articles,
    embed_articles,
    build_index as build_dense_index,
    search as dense_search,
)
from FlagEmbedding import BGEM3FlagModel

RRF_K = 60


def rrf_fuse(ranked_lists, k=RRF_K):
    """
    ranked_lists: list of lists, each an ordered list of article_ids
                  (best match first) from one retrieval method.
    Returns: list of (article_id, rrf_score) sorted by rrf_score desc.
    """
    scores = {}
    for ranked_ids in ranked_lists:
        for rank, article_id in enumerate(ranked_ids, start=1):
            scores[article_id] = scores.get(article_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def fused_search(query, bm25_index, bm25_articles, dense_collection, dense_model, k=15, pool=20):
    # Pull a slightly larger pool from each method (pool > k) before fusing,
    # so a document ranked e.g. #8 by BM25 but #1 by dense still has a
    # chance to surface in the fused top-k.
    bm25_hits = bm25_search(query, bm25_index, bm25_articles, k=pool)  # adjust to your bm25.py's real signature
    bm25_ids = [h["article_id"] for h in bm25_hits]

    dense_hits = dense_search(query, dense_collection, dense_model, k=pool)
    dense_ids = [h["article_id"] for h in dense_hits]

    fused = rrf_fuse([bm25_ids, dense_ids])
    return fused[:k]


if __name__ == "__main__":
    bm25_articles = load_bm25_articles()
    bm25_index = build_bm25_index(bm25_articles)

    dense_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    dense_articles = load_dense_articles()
    dense_vecs = embed_articles(dense_articles, dense_model)  # hits cache, instant
    dense_collection = build_dense_index(dense_articles, dense_vecs)

    smoke_queries = [
        "congé de maternité",
        "licenciement pour faute grave",
        "durée du travail heures supplémentaires",
    ]

    for q in smoke_queries:
        print(f"\n=== Query: {q} ===")
        results = fused_search(q, bm25_index, bm25_articles, dense_collection, dense_model)
        for article_id, score in results:
            print(f"  {article_id}  (rrf_score={score:.5f})")

        # --- Debug: inspect raw component lists for the maternité case ---
    print(f"\n{'='*60}")
    print("DEBUG: raw BM25 vs dense pools for 'congé de maternité'")
    print(f"{'='*60}")
    debug_bm25 = bm25_search("congé de maternité", bm25_index, bm25_articles, k=15)
    debug_dense = dense_search("congé de maternité", dense_collection, dense_model, k=15)
    print("BM25 pool article_ids:", [h["article_id"] for h in debug_bm25])
    print("Dense pool article_ids:", [h["article_id"] for h in debug_dense])
    print("Is 64 in BM25 pool?", "64" in [h["article_id"] for h in debug_bm25])