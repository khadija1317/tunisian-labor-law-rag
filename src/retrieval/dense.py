"""
src/retrieval/dense.py

Dense (semantic) retrieval for Corpus 1 (Code du Travail) using bge-m3.
Builds a persistent Chroma collection and exposes a search() function.
Complements bm25.py: where BM25 matches words, this matches meaning.
"""

import json
import time
from pathlib import Path

import numpy as np
import chromadb
from FlagEmbedding import BGEM3FlagModel

ARTICLES_PATH = Path("data/processed/code_du_travail_articles.json")
CHROMA_PATH = Path("data/chroma")
EMBEDDINGS_CACHE = Path("data/processed/dense_embeddings_cache.npy")
COLLECTION_NAME = "code_du_travail_dense"


def load_articles():
    with open(ARTICLES_PATH, encoding="utf-8") as f:
        articles = json.load(f)

    non_empty = [a for a in articles if a["text"].strip()]
    print(f"Loaded {len(articles)} articles, {len(non_empty)} non-empty "
          f"(skipping {len(articles) - len(non_empty)} repealed).")
    return non_empty


def embed_articles(articles, model):
    # Cache to disk: bge-m3 on CPU took ~9 min for 460 articles last run.
    # If anything downstream breaks, we re-load instead of re-embedding.
    if EMBEDDINGS_CACHE.exists():
        print(f"Loading cached embeddings from {EMBEDDINGS_CACHE}")
        return np.load(EMBEDDINGS_CACHE)

    texts = [a["text"] for a in articles]
    print(f"Embedding {len(texts)} articles with bge-m3 "
          f"(can take a few minutes on CPU)...")
    t0 = time.time()
    output = model.encode(
        texts,
        batch_size=12,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    dense_vecs = output["dense_vecs"]
    print(f"Done in {time.time() - t0:.1f}s")

    EMBEDDINGS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_CACHE, dense_vecs)
    print(f"Cached embeddings to {EMBEDDINGS_CACHE}")
    return dense_vecs


def build_index(articles, dense_vecs, client=None):
    if client is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [a["text"] for a in articles]
    ids = [a["article_id"] for a in articles]
    # Chroma metadata values must be str/int/float/bool -- None is
    # rejected outright, so coalesce missing/None amendment_note to "".
    metadatas = [{"amendment_note": a.get("amendment_note") or ""} for a in articles]

    collection.add(
        ids=ids,
        embeddings=dense_vecs.tolist(),
        documents=texts,
        metadatas=metadatas,
    )
    return collection


def search(query, collection, model, k=5):
    q_emb = model.encode(
        [query], return_dense=True, return_sparse=False, return_colbert_vecs=False
    )["dense_vecs"]
    results = collection.query(query_embeddings=q_emb.tolist(), n_results=k)
    hits = []
    for article_id, doc, dist in zip(
        results["ids"][0], results["documents"][0], results["distances"][0]
    ):
        hits.append({"article_id": article_id, "text": doc, "distance": dist})
    return hits


if __name__ == "__main__":
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

    articles = load_articles()
    dense_vecs = embed_articles(articles, model)
    collection = build_index(articles, dense_vecs)

    smoke_queries = [
        "congé de maternité",
        "licenciement pour faute grave",
        "durée du travail heures supplémentaires",
    ]

    for q in smoke_queries:
        print(f"\n=== Query: {q} ===")
        for hit in search(q, collection, model, k=3):
            print(f"  {hit['article_id']}  (distance={hit['distance']:.4f})  {hit['text'][:100]}...")