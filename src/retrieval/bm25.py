import json
import re
from rank_bm25 import BM25Okapi

ARTICLES_PATH = "data/processed/code_du_travail_articles.json"


def tokenize(text: str) -> list[str]:
    """Simple French-aware tokenizer: lowercase, split on non-alphanumeric
    (keeps accented chars and digits), treats apostrophes as word
    boundaries. Digits matter here: article numbers, durations (48 heures),
    percentages, and thresholds carry real legal meaning in this corpus --
    dropping them silently made BM25 blind to every numeric query."""
    text = text.lower()
    text = text.replace("’", "'")
    tokens = re.findall(r"[a-zàâäéèêëïîôöùûüç0-9]+", text)
    return tokens


def load_articles():
    with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
        articles = json.load(f)
    # Filter empty/repealed articles to match dense.py's indexed set
    # (472 total, 12 empty -> 460) so BM25 and dense are ranking over
    # the exact same pool before fusion.
    non_empty = [a for a in articles if a["text"].strip()]
    return non_empty


def build_bm25_index(articles):
    corpus_tokens = [tokenize(a["text"]) for a in articles]
    return BM25Okapi(corpus_tokens)


def search(query, bm25_index, articles, k=5):
    query_tokens = tokenize(query)
    scores = bm25_index.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    hits = []
    for idx in top_indices:
        a = articles[idx]
        hits.append({"article_id": a["article_id"], "text": a["text"], "score": scores[idx]})
    return hits


if __name__ == "__main__":
    articles = load_articles()
    print(f"Loaded {len(articles)} articles")

    bm25_index = build_bm25_index(articles)
    print("BM25 index built.")

    test_queries = [
        "licenciement pour faute grave",
        "congé de maternité",
        "durée du travail heures supplémentaires",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query!r}")
        print(f"{'='*60}")
        for hit in search(query, bm25_index, articles, k=3):
            print(f"  Article {hit['article_id']} (score={hit['score']:.2f}): {hit['text'][:100]}...")