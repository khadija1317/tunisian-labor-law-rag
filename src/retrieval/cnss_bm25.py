import json
import re
from rank_bm25 import BM25Okapi

FAQ_PATH = "data/processed/cnss_faq_entries.json"


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = text.replace("’", "'")
    tokens = re.findall(r"[a-zàâäéèêëïîôöùûüç]+", text)
    return tokens


def load_entries():
    with open(FAQ_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return entries


def entry_search_text(entry):
    return f"{entry['question']}\n{entry['answer']}"


def build_bm25_index(entries):
    corpus_tokens = [tokenize(entry_search_text(e)) for e in entries]
    return BM25Okapi(corpus_tokens)


def search(query, bm25_index, entries, k=5):
    query_tokens = tokenize(query)
    scores = bm25_index.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    hits = []
    for idx in top_indices:
        e = entries[idx]
        hits.append({
            "id": e["id"],
            "question": e["question"],
            "answer": e["answer"],
            "category": e["category"],
            "score": scores[idx],
        })
    return hits


if __name__ == "__main__":
    entries = load_entries()
    print(f"Loaded {len(entries)} CNSS FAQ entries")

    bm25_index = build_bm25_index(entries)
    print("BM25 index built.")

    test_queries = [
        "pension de vieillesse délai",
        "prêt logement remboursement",
        "prêt voiture conjoint",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query!r}")
        print(f"{'='*60}")
        for hit in search(query, bm25_index, entries, k=3):
            print(f"  {hit['id']} (score={hit['score']:.2f}): {hit['question'][:80]}")