"""
src/agents/verifier.py

Verifier agent: checks whether the synthesizer's answer is grounded in
the retrieved source text it cited. Two tiers:
 1. citation-existence check (free, structural, no LLM call)
 2. entailment check (LLM call) -- only runs if citations exist, since
    a fabricated citation is an automatic fail regardless of content.
"""

from dotenv import load_dotenv
load_dotenv()
from src.agents.errors import LLMCallError
from groq import Groq
from pydantic import BaseModel
from typing import List

MODEL_NAME = "openai/gpt-oss-20b"
client = Groq()


class VerifierOutput(BaseModel):
    grounded: bool
    flagged_claims: List[str]
    reasoning: str


def check_citations_exist(citations, retrieved_results, id_key="article_id"):
    valid_ids = [r[id_key] for r in retrieved_results]
    fabricated = [c for c in citations if c not in valid_ids]
    return len(fabricated) == 0, fabricated


def _format_cited_sources(citations, retrieved_results, id_key="article_id"):
    by_id = {r[id_key]: r for r in retrieved_results}
    blocks = []
    for c in citations:
        r = by_id.get(c)
        if r is None:
            continue
        if id_key == "article_id":
            blocks.append(f"[{c}]\n{r['text']}")
        else:
            blocks.append(f"[{c}] Q: {r['question']}\nR: {r['answer']}")
    return "\n\n".join(blocks)


SYSTEM_PROMPT = """Tu es un vérificateur de fidélité factuelle pour un assistant
juridique tunisien. On te donne une réponse générée et les extraits de sources
qui ont été cités pour la produire.

Vérifie si CHAQUE affirmation factuelle de la réponse est directement
soutenue par le texte des sources fournies. Une affirmation est "non
fondée" si elle ajoute une information, un chiffre, ou une exception qui
n'est pas explicitement dans les sources, même si cela semble plausible.

Les reformulations gardant le même sens (ex: "30 jours" et "environ un
mois") sont acceptées. Les changements de valeurs (ex: "30 jours" devient
"45 jours") ne sont PAS acceptés.

Réponds UNIQUEMENT avec un objet JSON de la forme :
{"grounded": true/false, "flagged_claims": ["..."], "reasoning": "..."}
"grounded" est true seulement si TOUTES les affirmations sont soutenues.
Aucun autre texte avant ou après le JSON."""


def verify_answer(query, synth_output, retrieved_results, corpus_type, max_attempts=2):
    id_key = "article_id" if corpus_type == "labor_code" else "id"

    citations_ok, fabricated = check_citations_exist(
        synth_output.citations, retrieved_results, id_key=id_key
    )
    if not citations_ok:
        return VerifierOutput(
            grounded=False,
            flagged_claims=[f"Citation inexistante: {c}" for c in fabricated],
            reasoning="Une ou plusieurs citations ne correspondent à aucune source récupérée.",
        )

    if not synth_output.citations:
        return VerifierOutput(
            grounded=False,
            flagged_claims=[],
            reasoning="Aucune citation fournie -- réponse non vérifiable.",
        )

    sources_text = _format_cited_sources(synth_output.citations, retrieved_results, id_key=id_key)
    user_message = f"""Question : {query}

Réponse générée : {synth_output.answer}

Sources citées :
{sources_text}"""

    last_error = None
    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            raw = response.choices[0].message.content
            return VerifierOutput.model_validate_json(raw)
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            last_error = e
            continue

    raise LLMCallError(f"Verifier failed after {max_attempts} attempts: {last_error!r}")



if __name__ == "__main__":
    from src.retrieval.rerank import build_reranker, retrieve as retrieve_labor
    from src.retrieval.bm25 import load_articles, build_bm25_index
    from src.retrieval.dense import load_articles as load_dense_articles, embed_articles, build_index as build_dense_index
    from src.agents.synthesizer import synthesize
    from FlagEmbedding import BGEM3FlagModel

    bm25_articles = load_articles()
    bm25_index = build_bm25_index(bm25_articles)
    articles_by_id = {a["article_id"]: a for a in bm25_articles}

    dense_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    dense_articles = load_dense_articles()
    dense_vecs = embed_articles(dense_articles, dense_model)
    dense_collection = build_dense_index(dense_articles, dense_vecs)

    reranker = build_reranker()

    query = "congé de maternité"
    results = retrieve_labor(query, bm25_index, bm25_articles, dense_collection, dense_model, articles_by_id, reranker)
    synth_output = synthesize(query, results, corpus_type="labor_code")
    print("Answer:", synth_output.answer)
    print("Citations:", synth_output.citations)

    verdict = verify_answer(query, synth_output, results, corpus_type="labor_code")
    print("Grounded:", verdict.grounded)
    print("Flagged:", verdict.flagged_claims)
    print("Reasoning:", verdict.reasoning)