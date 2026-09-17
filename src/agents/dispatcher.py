"""
Orchestrates the full chain: route -> retrieve -> (floor check) ->
synthesize -> verify. All index-building lives in setup.py; this file
only coordinates calls between the agent/retrieval modules.
"""

from src.agents import setup
from src.agents.router import route_query
from src.agents.synthesizer import synthesize, SynthesizerOutput
from src.agents.verifier import verify_answer
from src.agents.errors import LLMCallError

from src.retrieval.rerank import retrieve as retrieve_labor_code
from src.retrieval.cnss_rerank import retrieve as retrieve_cnss


LABOR_CODE_FLOOR = 0.01
CNSS_FLOOR = 0.001

API_ERROR_ANSWER = "Une erreur technique est survenue. Merci de réessayer."


def handle_query(query: str):
    try:
        label = route_query(query)
    except LLMCallError as e:
        return {
            "query": query, "label": None, "status": "api_error",
            "final_answer": API_ERROR_ANSWER,
            "citations": [], "grounded": None, "flagged_claims": [],
            "reasoning": f"router: {e}", "top_score": None,
        }

    if label == "out_of_scope":
        return {
            "query": query, "label": label, "status": "out_of_scope",
            "final_answer": "Désolé, nous n'avons pas suffisamment de données pour répondre à cette question.",
            "citations": [], "grounded": None, "flagged_claims": [], "reasoning": None,
            "top_score": None,
        }

    if label == "labor_code":
        results = retrieve_labor_code(
            query, setup.labor_bm25_index, setup.labor_bm25_articles,
            setup.labor_dense_collection, setup.dense_model,
            setup.labor_articles_by_id, setup.reranker,
        )
        floor = LABOR_CODE_FLOOR
    else:  # cnss
        results = retrieve_cnss(
            query, setup.cnss_bm25_index, setup.cnss_bm25_entries,
            setup.cnss_dense_collection, setup.dense_model,
            setup.cnss_entries_by_id, setup.reranker,
        )
        floor = CNSS_FLOOR

    top_score = results[0]["rerank_score"] if results else 0.0
    if top_score < floor:
        return {
            "query": query, "label": label, "status": "low_confidence_retrieval",
            "final_answer": "Nous n'avons pas trouvé d'information suffisamment pertinente pour répondre à cette question.",
            "citations": [], "grounded": None, "flagged_claims": [], "reasoning": None,
            "top_score": top_score,
        }

    try:
        synth_output = synthesize(query, results, corpus_type=label)
    except LLMCallError as e:
        return {
            "query": query, "label": label, "status": "api_error",
            "final_answer": API_ERROR_ANSWER,
            "citations": [], "grounded": None, "flagged_claims": [],
            "reasoning": f"synthesizer: {e}", "top_score": top_score,
        }

    try:
        verdict = verify_answer(query, synth_output, results, corpus_type=label)
    except LLMCallError as e:
        return {
            "query": query, "label": label, "status": "api_error",
            "final_answer": API_ERROR_ANSWER,
            "citations": synth_output.citations, "grounded": None, "flagged_claims": [],
            "reasoning": f"verifier: {e}", "top_score": top_score,
        }

    if not verdict.grounded:
        return {
            "query": query, "label": label, "status": "ungrounded",
            "final_answer": "Nous avons trouvé des informations partiellement pertinentes, mais elles n'étaient pas suffisantes pour garantir une réponse fiable à cette question.",
            "citations": synth_output.citations,
            "grounded": verdict.grounded,
            "flagged_claims": verdict.flagged_claims,
            "reasoning": verdict.reasoning,
            "top_score": top_score,
        }

    return {
        "query": query, "label": label, "status": "answered",
        "final_answer": synth_output.answer,
        "citations": synth_output.citations,
        "grounded": verdict.grounded,
        "flagged_claims": verdict.flagged_claims,
        "reasoning": verdict.reasoning,
        "top_score": top_score,
    }


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# PART 1: happy-path smoke tests (all three branches)")
    print("#" * 60)

    happy_queries = [
        ("congé de maternité", "expect: labor_code, answered, grounded True"),
        ("pension de vieillesse délai", "expect: cnss, answered"),
        ("meilleurs restaurants à Tunis", "expect: out_of_scope"),
    ]
    for q, note in happy_queries:
        print(f"\n{'='*60}\nQuery: {q}   ({note})\n{'='*60}")
        result = handle_query(q)
        for k, v in result.items():
            print(f"  {k}: {v}")

    print("\n" + "#" * 60)
    print("# PART 2: deliberate failure-path tests")
    print("#" * 60)

    print(f"\n{'='*60}\nTest A: floor-triggering query\n{'='*60}")
    result = handle_query("allocation chômage CNSS")
    for k, v in result.items():
        print(f"  {k}: {v}")

    print(f"\n{'='*60}\nTest B: manually corrupted synthesizer output\n{'='*60}")
    query = "congé de maternité"
    results = retrieve_labor_code(
        query, setup.labor_bm25_index, setup.labor_bm25_articles,
        setup.labor_dense_collection, setup.dense_model,
        setup.labor_articles_by_id, setup.reranker,
    )
    real_output = synthesize(query, results, corpus_type="labor_code")
    print(f"  Real answer: {real_output.answer}")

    corrupted_output = SynthesizerOutput(
        answer=real_output.answer + " Ce congé est également valable pour les employés à temps partiel sans réduction de salaire.",
        citations=real_output.citations,
    )
    print("Replace worked:", real_output.answer != corrupted_output.answer)
    print(f"  Corrupted answer: {corrupted_output.answer}")

    verdict = verify_answer(query, corrupted_output, results, corpus_type="labor_code")
    print(f"  grounded: {verdict.grounded}  (expect: False)")
    print(f"  flagged_claims: {verdict.flagged_claims}")
    print(f"  reasoning: {verdict.reasoning}")