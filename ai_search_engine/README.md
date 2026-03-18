# Technical Documentation Semantic Search Engine

A production-ready RAG (Retrieval-Augmented Generation) application built on top of
**[Endee](https://github.com/endee-io/endee)** — a high-performance open-source vector
database — that turns a folder of PDF and Markdown files into a queryable knowledge base
with LLM-generated answers.

---

## Why This Project Matters

Technical documentation is dense, scattered across files, and hard to search with
keywords alone. This project solves that by:

- Converting every document chunk into a semantic embedding.
- Storing and indexing those embeddings in Endee for sub-millisecond retrieval.
- Grounding LLM answers in the retrieved context, eliminating hallucinations.

The result is a search engine that understands *meaning*, not just keywords.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                       │
│                                                                 │
│  PDF / MD / TXT                                                 │
│       │                                                         │
│       ▼                                                         │
│  Text Extraction  (PyMuPDF for PDF, plain read for MD/TXT)      │
│       │                                                         │
│       ▼                                                         │
│  Chunker  (512-char chunks, 64-char overlap)                    │
│       │                                                         │
│       ▼                                                         │
│  Embedder  (sentence-transformers / all-MiniLM-L6-v2, 384-dim) │
│       │                                                         │
│       ▼                                                         │
│  Endee Upsert  ──► Endee Vector DB (HNSW + cosine, port 8080)  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         QUERY PIPELINE                          │
│                                                                 │
│  User Query                                                     │
│       │                                                         │
│       ▼                                                         │
│  Embedder  (same model, same 384-dim space)                     │
│       │                                                         │
│       ▼                                                         │
│  Endee Search  ──► top-k chunks (HNSW ANN, cosine similarity)  │
│       │                                                         │
│       ▼                                                         │
│  Prompt Builder  (RAG prompt with retrieved context)            │
│       │                                                         │
│       ▼                                                         │
│  LLM  (Ollama / OpenAI)  ──► Grounded Answer                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## How Endee Is Used

| Endee Feature | How It Is Used Here |
|---|---|
| **HNSW vector index** | Stores 384-dim embeddings; serves ANN queries in milliseconds |
| **Cosine similarity** | Measures semantic closeness between query and document embeddings |
| **Payload storage** | Each vector carries `source`, `chunk_index`, and `text` metadata |
| **Payload filtering** | Optional: filter results by `source` file or any metadata field |
| **HTTP API** | All operations (create index, upsert, search) go through the REST API |
| **Docker deployment** | Endee runs as a single Docker container — zero external dependencies |
| **CPU-targeted builds** | AVX2/AVX512/NEON builds ensure maximum throughput on any hardware |

---

## Project Structure

```
ai_search_engine/
├── endee_client.py     # Thin HTTP wrapper for the Endee REST API
├── embedder.py         # Sentence-Transformers embedding helper (lazy-loaded)
├── ingest.py           # Ingestion pipeline: files → chunks → embeddings → Endee
├── search.py           # RAG pipeline: query → Endee → LLM → answer
├── sample_docs/        # Two sample Markdown files to demo the pipeline
│   ├── hnsw_overview.md
│   └── sparse_search.md
├── .env.example        # Environment variable template
└── requirements.txt    # Python dependencies
```

---

## Setup Instructions

### Step 1 — Start Endee

**Windows / any OS with Docker (recommended):**

```bash
docker run \
  --ulimit nofile=100000:100000 \
  -p 8080:8080 \
  -v ./endee-data:/data \
  --name endee-server \
  --restart unless-stopped \
  endeeio/endee-server:latest
```

Or build from source (Linux / macOS):

```bash
# From the repo root
chmod +x ./install.sh ./run.sh
./install.sh --release --avx2   # use --neon on Apple Silicon
./run.sh
```

Verify: `curl http://localhost:8080/api/v1/health`

---

### Step 2 — Install Python dependencies

```bash
cd ai_search_engine
pip install -r requirements.txt
```

---

### Step 3 — Configure environment

```bash
cp .env.example .env
# Edit .env — set LLM_BACKEND, OLLAMA_MODEL or OPENAI_API_KEY
```

For Ollama (free, local):
```bash
ollama pull llama3.2
```

---

### Step 4 — Ingest documents

```bash
# Ingest the bundled sample docs
python ingest.py --docs ./sample_docs --index tech_docs

# Or point at your own folder
python ingest.py --docs /path/to/your/docs --index my_index
```

---

### Step 5 — Search and get answers

```bash
python search.py --query "How does filtered HNSW work in Endee?"
python search.py --query "What is sparse vector search and when should I use it?"
python search.py --query "How does Endee handle CPU optimizations?"
```

Example output:

```
============================================================
Query : How does filtered HNSW work in Endee?
============================================================

Answer:
Endee's filtered HNSW uses a pre-filtering strategy. The payload filter is
evaluated first, producing a RoaringBitmap of valid document IDs. For large
result sets (>= 1,000 IDs), the bitmap is passed to HNSW via a
BitMapFilterFunctor. For small result sets (< 1,000 IDs), HNSW is bypassed
and a brute-force scan is performed directly on the valid vectors.

Sources (3):
  • hnsw_overview.md  (score: 0.9231)
  • sparse_search.md  (score: 0.6104)
  • hnsw_overview.md  (score: 0.5873)
```

---

## Using as a Module

```python
from search import answer

result = answer(
    query="What CPU instruction sets does Endee support?",
    index="tech_docs",
    top_k=5,
)
print(result["answer"])
for s in result["sources"]:
    print(s["source"], s["score"])
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ENDEE_URL` | `http://localhost:8080` | Endee server URL |
| `ENDEE_AUTH_TOKEN` | _(empty)_ | Auth token if server was started with one |
| `LLM_BACKEND` | `ollama` | `ollama` or `openai` |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `OPENAI_API_KEY` | _(empty)_ | Required when `LLM_BACKEND=openai` |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |

---

## Evaluation Notes

This repository is a fork of [endee-io/endee](https://github.com/endee-io/endee).
The AI search engine application lives in the `ai_search_engine/`
directory and is self-contained — it only communicates with Endee through its
documented HTTP API.

---

## License

This project is built on top of Endee, which is licensed under the
[Apache License 2.0](../LICENSE). The application code in `ai_search_engine/` is
also released under Apache 2.0.
