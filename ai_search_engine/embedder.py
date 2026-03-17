# Embedding helper — wraps sentence-transformers
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None
MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, fast, good quality
EMBEDDING_DIM = 384


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Return a list of float vectors for the given texts."""
    return get_model().encode(texts, convert_to_numpy=True).tolist()


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
