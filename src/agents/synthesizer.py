"""
src/agents/synthesizer.py

Synthesizer agent: given a query and retrieved passages (from either
corpus), drafts a cited French answer. Structured output so the
verifier can check citations against what was actually retrieved.
"""

from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from pydantic import BaseModel, ValidationError
from typing import List

MODEL_NAME = "openai/gpt-oss-20b"

client = Groq()  # reads GROQ_API_KEY from environment


class SynthesizerOutput(BaseModel):
    answer: str
    citations: List[str]


SYSTEM_PROMPT = """Tu es un assistant juridique tunisien. Réponds à la question
de l'utilisateur UNIQUEMENT à partir des extraits fournis dans le message.
N'invente rien qui n'est pas explicitement dans ces extraits.

Si les extraits ne permettent pas de répondre à la question, dis-le
clairement dans le champ "answer" (par exemple : "Les documents fournis
ne contiennent pas d'information suffisante pour répondre à cette
question.") et laisse "citations" vide.

Les citations doivent être les identifiants exacts donnés entre crochets
dans les extraits (ex: "114" ou "cnss_faq_0005"), rien d'autre.

Réponds UNIQUEMENT avec un objet JSON de la forme :
{"answer": "...", "citations": ["...", "..."]}
Aucun autre texte avant ou après."""


def _format_context(retrieved_results, corpus_type):
    blocks = []
    for r in retrieved_results:
        if corpus_type == "labor_code":
            blocks.append(f"[{r['article_id']}]\n{r['text']}")
        elif corpus_type == "cnss":
            blocks.append(f"[{r['id']}] Q: {r['question']}\nR: {r['answer']}")
    return "\n\n".join(blocks)


def synthesize(query: str, retrieved_results, corpus_type: str, max_attempts: int = 2) -> SynthesizerOutput:
    context_text = _format_context(retrieved_results, corpus_type)
    user_message = f"Extraits disponibles :\n{context_text}\n\nQuestion : {query}"

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
            return SynthesizerOutput.model_validate_json(raw)
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            continue

    return SynthesizerOutput(
        answer="Une erreur est survenue lors de la génération de la réponse.",
        citations=[],
    )


if __name__ == "__main__":
    from src.retrieval.rerank import build_reranker, retrieve as retrieve_labor
    from src.retrieval.bm25 import load_articles, build_bm25_index
    from src.retrieval.dense import load_articles as load_dense_articles, embed_articles, build_index as build_dense_index
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

    output = synthesize(query, results, corpus_type="labor_code")
    print("Answer:", output.answer)
    print("Citations:", output.citations)