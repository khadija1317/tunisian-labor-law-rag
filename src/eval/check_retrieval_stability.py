from src.retrieval.rerank import retrieve as retrieve_labor_code
from src.agents import setup

query = "congé de maternité"

for run_num in range(1, 6):
    results = retrieve_labor_code(
        query, setup.labor_bm25_index, setup.labor_bm25_articles,
        setup.labor_dense_collection, setup.dense_model,
        setup.labor_articles_by_id, setup.reranker,
    )

    print(f"--- run {run_num} ---")
    for r in results:
        print(f"  article_id={r['article_id']}  rerank_score={r['rerank_score']}")
    print()