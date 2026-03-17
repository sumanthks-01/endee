# HNSW Algorithm in Endee

## Overview

Endee uses the Hierarchical Navigable Small World (HNSW) algorithm for approximate
nearest-neighbor (ANN) vector search. HNSW builds a multi-layer proximity graph
where each node represents a stored vector. During search, the algorithm enters the
graph at the top layer and greedily navigates toward the query vector, descending
through layers until it reaches the bottom layer where the final candidates are
collected.

## Filtered HNSW

When a query includes payload filters, Endee applies a pre-filtering strategy:

1. The filter (e.g., `category = "networking"`) is evaluated first using the
   category or numeric index, producing a RoaringBitmap of valid document IDs.
2. If the result set is large (>= 1,000 IDs), the bitmap is passed to HNSW via
   a `BitMapFilterFunctor`, which skips invalid nodes during graph traversal.
3. If the result set is small (< 1,000 IDs), HNSW is bypassed entirely and a
   brute-force distance scan is performed directly on the valid vectors.

This adaptive strategy ensures optimal latency regardless of filter selectivity.

## CPU Optimizations

Endee compiles distance kernels for AVX2, AVX512, NEON, and SVE2 instruction sets.
The correct kernel is selected at build time via CMake flags, enabling the CPU to
process multiple float lanes in a single instruction cycle.

## Index Parameters

- `ef_construction`: Controls graph quality during build. Higher = better recall,
  slower inserts.
- `M`: Number of bidirectional links per node. Higher = better recall, more memory.
- `ef_search`: Beam width during query. Higher = better recall, slower queries.
