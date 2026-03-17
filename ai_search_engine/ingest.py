"""
ingest.py — Process a folder of PDF / Markdown files and upsert them into Endee.

Usage:
    python ingest.py --docs ./sample_docs --index tech_docs

Pipeline:
    Raw file → text extraction → fixed-size chunks → embeddings → Endee insert
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import fitz  # PyMuPDF
from tqdm import tqdm

from embedder import EMBEDDING_DIM, embed
from endee_client import EndeeClient

# ── tunables ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = 512        # characters per chunk
CHUNK_OVERLAP = 64      # overlap between consecutive chunks
INSERT_BATCH = 32       # vectors per HTTP request


# ── text extraction ───────────────────────────────────────────────────────────
def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        with fitz.open(path) as doc:
            return "\n".join(page.get_text() for page in doc)
    return path.read_text(encoding="utf-8", errors="ignore")


# ── chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str) -> list[str]:
    """Split text into overlapping fixed-size character chunks."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c.strip()) > 30]


# ── stable ID ─────────────────────────────────────────────────────────────────
def chunk_id(source: str, idx: int) -> str:
    return hashlib.md5(f"{source}::{idx}".encode()).hexdigest()


# ── main ──────────────────────────────────────────────────────────────────────
def ingest(docs_dir: str, index_name: str, endee_url: str, auth_token: str) -> None:
    client = EndeeClient(endee_url, auth_token)

    client.health()
    print("✓ Endee server is healthy")

    if not client.index_exists(index_name):
        # Endee API requires: index_name, dim, space_type
        client.create_index(index_name, EMBEDDING_DIM, space_type="cosine")
        print(f"✓ Created index '{index_name}' (dim={EMBEDDING_DIM}, space_type=cosine)")
    else:
        print(f"✓ Index '{index_name}' already exists — inserting into it")

    supported = {".pdf", ".md", ".txt"}
    files = [p for p in Path(docs_dir).rglob("*") if p.suffix.lower() in supported]
    if not files:
        print(f"No PDF/Markdown/TXT files found in '{docs_dir}'")
        return

    batch: list[dict] = []
    total_chunks = 0

    for file in tqdm(files, desc="Ingesting files"):
        text = extract_text(file)
        chunks = chunk_text(text)
        vectors = embed(chunks)

        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            # Endee stores payload as:
            #   "meta"   — free-form string (we JSON-encode our metadata here)
            #   "filter" — JSON-encoded object used for payload filtering queries
            meta = json.dumps({"text": chunk, "chunk_index": idx})
            filter_str = json.dumps({"source": file.name})

            batch.append({
                "id": chunk_id(file.name, idx),
                "vector": vector,
                "meta": meta,
                "filter": filter_str,
            })

            if len(batch) >= INSERT_BATCH:
                client.insert(index_name, batch)
                total_chunks += len(batch)
                batch = []

    if batch:
        client.insert(index_name, batch)
        total_chunks += len(batch)

    print(f"\n✓ Ingested {total_chunks} chunks from {len(files)} file(s) into '{index_name}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Endee")
    parser.add_argument("--docs",  default="./sample_docs", help="Folder with PDF/MD/TXT files")
    parser.add_argument("--index", default="tech_docs",     help="Endee index name")
    parser.add_argument("--url",   default=os.getenv("ENDEE_URL", "http://localhost:8080"))
    parser.add_argument("--token", default=os.getenv("ENDEE_AUTH_TOKEN", ""))
    args = parser.parse_args()

    ingest(args.docs, args.index, args.url, args.token)
