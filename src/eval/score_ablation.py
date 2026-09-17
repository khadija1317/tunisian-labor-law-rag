"""
src/eval/score_ablation.py

Verifier ablation: runs route -> retrieve -> (floor) -> synthesize for
each gold pair, WITHOUT calling verify_answer(). Reports what a 3-agent
architecture (no verifier) would have shown the user, for direct
comparison against the actual 4-agent (gated) result.

The case that matters: does any bucket-B (unanswerable-in-domain) pair
come back status=="answered" here? If yes, that's a confident wrong
answer the verifier is the only thing stopping. If no bucket-B pair
slips through, the synthesizer is already hedging on its own and the
verifier's marginal value is smaller than assumed -- also a real
finding, just a different one.
"""

import json
from src.agents import setup
from src.agents.router import route_query
from src.agents.synthesizer import synthesize
from src.agents.errors import LLMCallError
from src.retrieval.rerank import retrieve as retrieve_labor_code
from src.retrieval.cnss_rerank import retrieve as retrieve_cnss

LABOR_CODE_FLOOR = 0.01
CNSS_FLOOR = 0.001

BUCKET_B = {"g008", "g014", "g015", "g016", "g019"}

with open("data/eval/qa_pairs.json", encoding="utf-8") as f:
    gold_pairs = json.load(f)


def no_verifier_pipeline(query):
    try:
        label = route_query(query)
    except LLMCallError as e:
        return {"status": "api_error", "detail": f"router: {e}"}

    if label == "out_of_scope":
        return {"status": "out_of_scope", "answer": None, "citations": []}

    if label == "labor_code":
        results = retrieve_labor_code(
            query, setup.labor_bm25_index, setup.labor_bm25_articles,
            setup.labor_dense_collection, setup.dense_model,
            setup.labor_articles_by_id, setup.reranker,
        )
        floor = LABOR_CODE_FLOOR
    else:
        results = retrieve_cnss(
            query, setup.cnss_bm25_index, setup.cnss_bm25_entries,
            setup.cnss_dense_collection, setup.dense_model,
            setup.cnss_entries_by_id, setup.reranker,
        )
        floor = CNSS_FLOOR

    top_score = results[0]["rerank_score"] if results else 0.0
    if top_score < floor:
        return {"status": "low_confidence_retrieval", "answer": None, "citations": []}

    try:
        synth_output = synthesize(query, results, corpus_type=label)
    except LLMCallError as e:
        return {"status": "api_error", "detail": f"synthesizer: {e}"}

    # No-verifier architecture: whatever the synthesizer produced IS
    # what the user would see, unfiltered.
    return {"status": "answered", "answer": synth_output.answer,
            "citations": synth_output.citations}


print(f"{'id':6} {'no-verifier status':22} {'citations':30} answer_preview")
print("-" * 100)

flagged_passthrough = []

for pair in gold_pairs:
    pid = pair["id"]
    result = no_verifier_pipeline(pair["query"])
    status = result["status"]
    citations = result.get("citations", [])
    answer_preview = (result.get("answer") or "")[:70]

    print(f"{pid:6} {status:22} {str(citations):30} {answer_preview}")

    if pid in BUCKET_B and status == "answered":
        flagged_passthrough.append(pid)

print("\n" + "=" * 50)
if flagged_passthrough:
    print(f"WITHOUT the verifier, these unanswerable-in-domain pairs "
          f"would have reached the user as confident answers: {flagged_passthrough}")
else:
    print("No bucket-B pairs slipped through as 'answered' even without "
          "the verifier -- the synthesizer itself hedged on all of them.")