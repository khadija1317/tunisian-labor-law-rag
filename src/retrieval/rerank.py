"""

Cross-encoder reranking (bge-reranker-v2-m3) over a fused BM25+dense
candidate pool.
"""

from FlagEmbedding import FlagReranker
import time

def build_reranker():
    return FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=False)


def rerank(query, fused_results, articles_by_id, reranker, top_n=5):

    pairs = []
    meta = []
    for rrf_rank, (article_id, rrf_score) in enumerate(fused_results, start=1):
        article = articles_by_id[article_id]
        pairs.append([query, article["text"]])
        meta.append({
            "article_id": article_id,
            "rrf_score": rrf_score,
            "rrf_rank": rrf_rank,
            "text": article["text"],
        })


    scores = reranker.compute_score(pairs, normalize=True)
    if isinstance(scores, float):  # single-pair edge case
        scores = [scores]

    for m, s in zip(meta, scores):
        m["rerank_score"] = s

    return sorted(meta, key=lambda x: x["rerank_score"], reverse=True)[:top_n]







def retrieve(query, bm25_index, bm25_articles, dense_collection, dense_model,
             articles_by_id, reranker, pool=20, rrf_k=15, top_n=5):

    fused = fused_search(query, bm25_index, bm25_articles, dense_collection,
                          dense_model, k=rrf_k, pool=pool)
    return rerank(query, fused, articles_by_id, reranker, top_n=top_n)


if __name__ == "__main__":
    from src.retrieval.bm25 import load_articles as load_bm25_articles, build_bm25_index
    from src.retrieval.dense import (
        load_articles as load_dense_articles,
        embed_articles,
        build_index as build_dense_index,
    )
    from src.retrieval.fusion import fused_search
    from FlagEmbedding import BGEM3FlagModel

    bm25_articles = load_bm25_articles()
    bm25_index = build_bm25_index(bm25_articles)
    articles_by_id = {a["article_id"]: a for a in bm25_articles}

    dense_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    dense_articles = load_dense_articles()
    dense_vecs = embed_articles(dense_articles, dense_model)
    dense_collection = build_dense_index(dense_articles, dense_vecs)

    reranker = build_reranker()

    smoke_queries = [
        "congé de maternité",
        "licenciement pour faute grave",
        "durée du travail heures supplémentaires",
    ]

    for q in smoke_queries:
        print(f"\n=== Query: {q} ===")


        t0 = time.time()
        fused = fused_search(q, bm25_index, bm25_articles, dense_collection, dense_model, k=15, pool=20)
        t1 = time.time()
        reranked = rerank(q, fused, articles_by_id, reranker, top_n=5)
        t2 = time.time()

        print(f"  [retrieval: {(t1-t0)*1000:.0f}ms]  [rerank: {(t2-t1)*1000:.0f}ms]  [pairs scored: {len(fused)}]")
        for r in reranked:
            print(f"  {r['article_id']}  rerank={r['rerank_score']:.4f}  (was rrf_rank={r['rrf_rank']}, rrf_score={r['rrf_score']:.5f})")
        