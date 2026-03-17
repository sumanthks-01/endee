# Sparse Vector Search in Endee

## What Is Sparse Search?

Sparse vectors represent text as high-dimensional vectors where most values are zero.
Each non-zero dimension corresponds to a term (word or sub-word token) and its weight
(e.g., TF-IDF or BM25 score). Sparse search computes dot-product similarity between
the query sparse vector and stored document sparse vectors.

## Hybrid Search

Endee supports hybrid retrieval by combining dense vector search (semantic similarity
via embeddings) with sparse vector search (term-level precision). The scores from both
modalities can be fused using Reciprocal Rank Fusion (RRF) or a weighted sum.

Use hybrid search when:
- Exact keyword matches matter (e.g., product codes, error codes, proper nouns).
- Semantic search alone misses rare or domain-specific terms.

## Storage Architecture

Sparse vectors are stored in an inverted index backed by MDBX (an embedded key-value
store). Each term maps to a posting list of (document_id, weight) pairs, organized
into fixed-size blocks of up to 65,535 entries. Weights are quantized to uint8 by
default to reduce storage and improve cache efficiency.

## API Usage

To store a sparse vector alongside a dense vector, include a `sparse_vector` field
in the upsert payload:

```json
{
  "id": "doc_001",
  "vector": [0.12, -0.34, ...],
  "sparse_vector": {"indices": [42, 1337, 9001], "values": [0.8, 0.5, 0.3]},
  "payload": {"source": "manual.pdf"}
}
```

## Performance

The search path uses a batch scoring loop with a dense accumulation buffer and a
min-heap for top-k extraction. SIMD helpers (AVX2/AVX512/NEON/SVE2) accelerate
tombstone skipping and doc-id lookup within blocks.
