import json
from src.agents.dispatcher import handle_query

with open("data/eval/qa_pairs.json", "r", encoding="utf-8") as f:
    gold_pairs = json.load(f)

for pair in gold_pairs:
    result = handle_query(pair["query"])

    print(f"--- {pair['id']} ---")
    print("query:", pair["query"])
    print("status:", result["status"], "  top_score:", result["top_score"])
    print("expected route:", pair["expected_route_label"], "| actual:", result["label"])
    print("expected citations:", pair["expected_citations"], "| actual:", result["citations"])
    print("expected grounded:", pair["expected_grounded"], "| actual:", result["grounded"])
    if result["status"] == "api_error":
        print("!! ERROR DETAIL:", result["reasoning"])
    print()