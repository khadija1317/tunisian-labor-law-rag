"""
Dispatches a query to the correct corpus retriever based on router.py's
classification. Builds all retrieval resources ONCE at import time — the
embedding model and reranker are shared across both corpora (per Day 4's
design note that the cross-encoder isn't corpus-specific).
"""

from FlagEmbedding import BGEM3FlagModel

from src.agents.router import route_query

from src.retrieval.bm25 import load_articles as load_labor_bm25_articles, build_bm25_index as build_labor_bm25_index
from src.retrieval.dense import (
    load_articles as load_labor_dense_articles,
    embed_articles,
    build_index as build_labor_dense_index,
)
from src.retrieval.rerank import build_reranker, retrieve as retrieve_labor_code

from src.retrieval.cnss_bm25 import load_entries as load_cnss_bm25_entries, build_bm25_index as build_cnss_bm25_index
from src.retrieval.cnss_dense import (
    load_entries as load_cnss_dense_entries,
    embed_entries,
    build_index as build_cnss_dense_index,
)
from src.retrieval.cnss_rerank import retrieve as retrieve_cnss

# --- One-time setup: shared model + reranker, then per-corpus indices ---

print("Loading shared dense model + reranker...")
_dense_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
_reranker = build_reranker()

print("Building labor code index...")
_labor_bm25_articles = load_labor_bm25_articles()
_labor_bm25_index = build_labor_bm25_index(_labor_bm25_articles)
_labor_articles_by_id = {a["article_id"]: a for a in _labor_bm25_articles}
_labor_dense_articles = load_labor_dense_articles()
_labor_dense_vecs = embed_articles(_labor_dense_articles, _dense_model)
_labor_dense_collection = build_labor_dense_index(_labor_dense_articles, _labor_dense_vecs)

print("Building CNSS index...")
_cnss_bm25_entries = load_cnss_bm25_entries()
_cnss_bm25_index = build_cnss_bm25_index(_cnss_bm25_entries)
_cnss_entries_by_id = {e["id"]: e for e in _cnss_bm25_entries}
_cnss_dense_entries = load_cnss_dense_entries()
_cnss_dense_vecs = embed_entries(_cnss_dense_entries, _dense_model)
_cnss_dense_collection = build_cnss_dense_index(_cnss_dense_entries, _cnss_dense_vecs)

print("Dispatcher ready.")


def handle_query(query: str):
    label = route_query(query)

    if label == "labor_code":
        return retrieve_labor_code(
            query, _labor_bm25_index, _labor_bm25_articles,
            _labor_dense_collection, _dense_model, _labor_articles_by_id, _reranker,
        )
    elif label == "cnss":
        return retrieve_cnss(
            query, _cnss_bm25_index, _cnss_bm25_entries,
            _cnss_dense_collection, _dense_model, _cnss_entries_by_id, _reranker,
        )
    else:
        return "Désolé, nous n'avons pas suffisamment de données pour répondre à cette question."