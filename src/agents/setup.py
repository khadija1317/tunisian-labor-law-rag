"""
src/agents/setup.py

One-time setup: builds the shared dense model, reranker, and both
corpora's retrieval indices. Pulled out of dispatcher.py so dispatcher
can focus purely on orchestration (route -> retrieve -> synthesize ->
verify), not initialization.
"""
import chromadb
from pathlib import Path


from FlagEmbedding import BGEM3FlagModel

from src.retrieval.bm25 import load_articles as load_labor_bm25_articles, build_bm25_index as build_labor_bm25_index
from src.retrieval.dense import (
    load_articles as load_labor_dense_articles,
    embed_articles,
    build_index as build_labor_dense_index,
)
from src.retrieval.rerank import build_reranker

from src.retrieval.cnss_bm25 import load_entries as load_cnss_bm25_entries, build_bm25_index as build_cnss_bm25_index
from src.retrieval.cnss_dense import (
    load_entries as load_cnss_dense_entries,
    embed_entries,
    build_index as build_cnss_dense_index,
)
CHROMA_PATH = Path("data/chroma")
_shared_chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))


print("Loading shared dense model + reranker...")
dense_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
reranker = build_reranker()

print("Building labor code index...")
labor_bm25_articles = load_labor_bm25_articles()
labor_bm25_index = build_labor_bm25_index(labor_bm25_articles)
labor_articles_by_id = {a["article_id"]: a for a in labor_bm25_articles}
labor_dense_articles = load_labor_dense_articles()
labor_dense_vecs = embed_articles(labor_dense_articles, dense_model)
labor_dense_collection = build_labor_dense_index(labor_dense_articles, labor_dense_vecs, client=_shared_chroma_client)

print("Building CNSS index...")
cnss_bm25_entries = load_cnss_bm25_entries()
cnss_bm25_index = build_cnss_bm25_index(cnss_bm25_entries)
cnss_entries_by_id = {e["id"]: e for e in cnss_bm25_entries}
cnss_dense_entries = load_cnss_dense_entries()
cnss_dense_vecs = embed_entries(cnss_dense_entries, dense_model)
cnss_dense_collection = build_cnss_dense_index(cnss_dense_entries, cnss_dense_vecs, client=_shared_chroma_client)

print("Setup ready.")