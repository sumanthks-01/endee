"""
search.py — Semantic search + RAG generation against an Endee index.

Usage (CLI):
    python search.py --query "How does HNSW filtering work?" --index tech_docs

Usage (as a module):
    from search import answer
    response = answer("What is sparse vector search?")
"""
import argparse
import json
import os

from embedder import embed_one
from endee_client import EndeeClient

# ── LLM backend selection ─────────────────────────────────────────────────────
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _generate_openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def _generate_ollama(prompt: str) -> str:
    import ollama
    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp["message"]["content"].strip()


def generate(prompt: str) -> str:
    if LLM_BACKEND == "openai":
        return _generate_openai(prompt)
    return _generate_ollama(prompt)


# ── helpers ───────────────────────────────────────────────────────────────────
def _parse_json_field(raw) -> dict:
    """
    Parse a JSON field that may be:
      - a plain JSON string:          '{"text": "..."}'
      - a Python bytes repr string:   "b'{\"text\": \"...\"}'"
    """
    if not raw:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    # Strip Python bytes repr prefix: b'...' or b"..."
    s = str(raw).strip()
    if s.startswith(("b'", 'b"')):
        s = s[2:-1]                          # drop b' prefix and trailing '
        s = s.replace("\\'", "'")            # unescape single quotes
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return {"text": s}


# ── RAG pipeline ──────────────────────────────────────────────────────────────
RAG_PROMPT = """\
You are a technical documentation assistant. Answer the question using ONLY the
context below. If the context does not contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""


def answer(
    query: str,
    index: str = "tech_docs",
    top_k: int = 5,
    endee_url: str = "http://localhost:8080",
    auth_token: str = "",
) -> dict:
    """
    Full RAG pipeline:
      1. Embed the query.
      2. Retrieve top-k chunks from Endee.
      3. Decode meta / filter fields and build grounded prompt.
      4. Call the LLM and return the answer.
    """
    client = EndeeClient(endee_url, auth_token)
    vector = embed_one(query)
    results = client.search(index, vector, top_k=top_k)

    if not results:
        return {"query": query, "sources": [], "answer": "No relevant documents found."}

    context_parts = []
    sources = []

    for r in results:
        meta        = _parse_json_field(r.get("meta", ""))
        filter_data = _parse_json_field(r.get("filter", ""))

        text   = meta.get("text", "")
        source = filter_data.get("source", "unknown")
        score  = float(r.get("score", 0.0))

        context_parts.append(f"[{source}] {text}")
        sources.append({"source": source, "score": round(score, 4)})

    context = "\n\n".join(context_parts)
    prompt = RAG_PROMPT.format(context=context, question=query)
    llm_answer = generate(prompt)

    return {"query": query, "sources": sources, "answer": llm_answer}


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic search + RAG over Endee")
    parser.add_argument("--query", required=True, help="Natural language question")
    parser.add_argument("--index", default="tech_docs")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--url",   default=os.getenv("ENDEE_URL", "http://localhost:8080"))
    parser.add_argument("--token", default=os.getenv("ENDEE_AUTH_TOKEN", ""))
    args = parser.parse_args()

    result = answer(args.query, args.index, args.top_k, args.url, args.token)

    print(f"\n{'='*60}")
    print(f"Query : {result['query']}")
    print(f"{'='*60}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources ({len(result['sources'])}):")
    for s in result["sources"]:
        print(f"  • {s['source']}  (score: {s['score']})")
