# Tunisian Labor Law & CNSS Assistant


**Demo video:** _coming soon_



A RAG system answering questions about the Tunisian Code du Travail and CNSS (social security) procedures, with hybrid retrieval, cross-encoder reranking, and a groundedness verifier gating every answer.

## Overview

This project explores whether a RAG pipeline — with an explicit verification step — produces more trustworthy, interpretable answers than a single-pass retrieve-and-generate system, for a domain (Tunisian labor law) where an unsupported claim has real consequences.

**Two corpora:**
- **Code du Travail Tunisien** — 472 articles extracted from the official PDF 
- **CNSS FAQ** — 40 entries across 5 categories 

## Architecture


1. **Router** — single-label classification (`labor_code` / `cnss` / `out_of_scope`) via Groq (`openai/gpt-oss-20b`), structured output with Pydantic
2. **Retriever** — one independent pipeline per corpus: BM25 + dense (bge-m3) retrieval, fused via Reciprocal Rank Fusion, reranked with a cross-encoder (bge-reranker-v2-m3)
3. **Synthesizer** — drafts a French answer with explicit citations, grounded strictly in retrieved passages
4. **Verifier** — two-tier check before an answer is released:
   - **Tier 1** (free, no LLM call): do the cited sources actually exist in what was retrieved?
   - **Tier 2** (LLM entailment): does every claim in the answer hold up against its cited source?

If the verifier flags an answer as ungrounded, the system returns a fallback message rather than the unsupported answer — the failure mode is "I don't know," not a confident wrong answer.

## Stack

- **Retrieval:** `rank_bm25`, `chromadb`, `FlagEmbedding` (bge-m3 dense embeddings, bge-reranker-v2-m3 cross-encoder)
- **Agents:** Groq API (`openai/gpt-oss-20b`), `pydantic` for structured outputs
- **UI:** Gradio
- **Evaluation:** hand-written set + `ragas` (Faithfulness)

## Setup

```bash
git clone https://github.com/khadija1317/tunisian-labor-law-rag.git
cd tunisian-labor-law-rag
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root:
GROQ_API_KEY=your_key_here


Run the app:
```bash
python app.py
```

First run downloads the embedding/reranker models from Hugging Face (~4-5GB total) and, if no cached embeddings are found, embeds both corpora from scratch (~9 minutes on CPU). Subsequent runs load from cache.

**Note:** the app is not currently hosted publicly — see [Known Limitations](#known-limitations). Run locally, or see the accompanying demo video.

## Evaluation

21 hand-written gold Q&A pairs, bucketed by type rather than scored with a single blended metric:

| Bucket | Description | n | Metric |
|---|---|---|---|
| A | Single-source answerable | 13 | Recall@5 / MRR@5 (retrieval), exact citation + grounded=True (full-chain) |
| A2 | Known multi-source (conflicting duplicate FAQ entries) | 1 | Same as A, subset-match on full-chain |
| B | Unanswerable-in-domain | 5 | Full-chain pass = correctly flagged as ungrounded/low-confidence |
| C | Out-of-scope | 2 | Full-chain pass = correctly routed out-of-scope |

**Results:**
- Retrieval-only (Bucket A+A2, n=14): Recall@5 = 1.000, MRR@5 = 0.964
- Full-chain (all 21 pairs): 21/21 pass across all buckets
- RAGAS Faithfulness (Bucket A+A2, n=13 valid of 14): mean ≈ 0.92

## Known Limitations

- **Reranker latency:** the cross-encoder reranker runs fp32 on CPU and accounts for ~97% of per-query latency (~75-115s). This is an architecture cost of the current hardware, candidate pool size was tuned down as a partial mitigation.
- **Cross-sector retrieval bleed:** the reranker occasionally ranks a topically-similar article from the wrong sector (e.g. agricultural-sector provisions) above the correct general-sector article, even when the query explicitly excludes that sector.
- **CNSS FAQ coverage skew:** some procedural questions fall outside what the 40-entry FAQ corpus covers; the system correctly declines rather than fabricating an answer, but coverage is limited by source material.
- **No live deployment:** three free hosting paths were evaluated and ruled out for concrete technical reasons (memory limits, platform account restrictions, and unreliable free-tier compute availability). The app runs fully locally; see the demo video for a live walkthrough.
- **LLM-as-judge reliability:** the RAGAS evaluator occasionally misjudged a clearly-supported answer as unfaithful (reproduced across two independent runs on the same query).

## Project Structure

src/
├── orchestration/# router, synthesizer, verifier, dispatcher 
├── retrieval/ # BM25, dense, fusion, reranking — one path per corpus
└── eval/ # scoring scripts (retrieval, full-chain, ablation, latency, RAGAS)
data/
├── raw/ # source PDF/HTML
├── processed/ # extracted articles/entries, cached embeddings
├── chroma/ # vector store (rebuilt on startup)
└── eval/ # gold Q&A pairs, eval results
app.py # Gradio UI