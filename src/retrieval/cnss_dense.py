"""
Dense retrieval for Corpus 2 (CNSS FAQ) using bge-m3.
Mirrors dense.py's structure; search text = question+answer concatenated
(see cnss_bm25.entry_search_text).
"""

import json
import time
from pathlib import Path

import numpy as np
import chromadb
from FlagEmbedding import BGEM3FlagModel

from src.retrieval.cnss_bm25 import entry_search_text

FAQ_PATH = Path("data/processed/cnss_faq_entries.json")
CHROMA_PATH = Path("data/chroma")
EMBEDDINGS_CACHE = Path("data/processed/cnss_faq_dense_embeddings_cache.npy")
COLLECTION_NAME = "cnss_faq_dense"


def load_entries():
    with open(FAQ_PATH, encoding="utf-8") as f:
        entries = json.load(f)
    print(f"Loaded {len(entries)} CNSS FAQ entries.")
    return entries


def embed_entries(entries, model):
    if EMBEDDINGS_CACHE.exists():
        cached = np.load(EMBEDDINGS_CACHE)
        if cached.shape[0] == len(entries):
            print(f"Loading cached embeddings from {EMBEDDINGS_CACHE}")
            return cached
        print(
            f"Cache has {cached.shape[0]} rows but corpus has {len(entries)} "
            f"entries -- stale cache, re-embedding."
        )

    texts = [entry_search_text(e) for e in entries]
    print(f"Embedding {len(texts)} CNSS entries with bge-m3...")
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


def build_index(entries, dense_vecs, client=None):
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

    texts = [entry_search_text(e) for e in entries]
    ids = [e["id"] for e in entries]
    metadatas = [{"category": e["category"], "question": e["question"]} for e in entries]

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
    for entry_id, doc, meta, dist in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({
            "id": entry_id,
            "question": meta["question"],
            "category": meta["category"],
            "text": doc,
            "distance": dist,
        })
    return hits


if __name__ == "__main__":
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

    entries = load_entries()
    dense_vecs = embed_entries(entries, model)
    collection = build_index(entries, dense_vecs)

    smoke_queries = [
        "pension de vieillesse délai",
        "prêt logement remboursement",
        "prêt voiture conjoint",
    ]

    for q in smoke_queries:
        print(f"\n=== Query: {q} ===")
        for hit in search(q, collection, model, k=3):
            print(f"  {hit['id']}  (distance={hit['distance']:.4f})  {hit['question'][:80]}")