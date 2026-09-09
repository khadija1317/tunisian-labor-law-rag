import json
import re
from rank_bm25 import BM25Okapi

ARTICLES_PATH = "data/processed/code_du_travail_articles.json"


def tokenize(text: str) -> list[str]:
    """Simple French-aware tokenizer: lowercase, split on non-alphanumeric
    (keeps accented chars), treats apostrophes as word boundaries."""
    text = text.lower()
    text = text.replace("’", "'")  # normalize curly apostrophe to straight
    tokens = re.findall(r"[a-zàâäéèêëïîôöùûüç]+", text)
    return tokens


with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
    articles = json.load(f)

print(f"Loaded {len(articles)} articles")

# Build the corpus: tokenize each article's text
corpus_tokens = [tokenize(a["text"]) for a in articles]

bm25 = BM25Okapi(corpus_tokens)

print("BM25 index built.")

# --- Smoke test ---
test_queries = [
    "licenciement pour faute grave",
    "congé de maternité",
    "durée du travail heures supplémentaires",
]

for query in test_queries:
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]

    print(f"\n{'='*60}")
    print(f"Query: {query!r}")
    print(f"{'='*60}")
    for idx in top_indices:
        a = articles[idx]
        print(f"  Article {a['article_id']} (score={scores[idx]:.2f}): {a['text'][:100]}...")