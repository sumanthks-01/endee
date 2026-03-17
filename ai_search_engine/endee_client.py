# Endee client — thin wrapper around the Endee HTTP API
import json

import msgpack
import requests


class EndeeClient:
    """Minimal HTTP client for the Endee vector database."""

    def __init__(self, base_url: str = "http://localhost:8080", auth_token: str = ""):
        self.base = base_url.rstrip("/")
        self.headers = {"Authorization": auth_token} if auth_token else {}

    def _url(self, path: str) -> str:
        return f"{self.base}{path}"

    # ------------------------------------------------------------------ index
    def create_index(self, name: str, dim: int, space_type: str = "cosine") -> str:
        """Required fields per Endee API: index_name, dim, space_type."""
        payload = {"index_name": name, "dim": dim, "space_type": space_type}
        r = requests.post(self._url("/api/v1/index/create"), json=payload, headers=self.headers)
        r.raise_for_status()
        return r.text

    def index_exists(self, name: str) -> bool:
        r = requests.get(self._url("/api/v1/index/list"), headers=self.headers)
        r.raise_for_status()
        # Response: {"indexes": [{"name": "admin/index_name", ...}, ...]}
        indexes = r.json().get("indexes", [])
        return any(idx.get("name", "").endswith(f"/{name}") for idx in indexes)

    # ----------------------------------------------------------------- insert
    def insert(self, index: str, vectors: list[dict]) -> None:
        """
        vectors: list of {"id": str, "vector": [...], "meta": str, "filter": str}
        Endpoint: POST /api/v1/index/<index>/vector/insert
        """
        url = self._url(f"/api/v1/index/{index}/vector/insert")
        headers = {**self.headers, "Content-Type": "application/json"}
        r = requests.post(url, data=json.dumps(vectors), headers=headers)
        r.raise_for_status()

    # ------------------------------------------------------------------ search
    def search(self, index: str, vector: list[float], top_k: int = 5) -> list[dict]:
        """
        Endpoint: POST /api/v1/index/<index>/search
        Response: MessagePack list of [score, id, meta_str, filter_str, ?, []]
        Returns normalised list of {"score", "id", "meta", "filter"} dicts.
        """
        payload = {"vector": vector, "k": top_k}
        url = self._url(f"/api/v1/index/{index}/search")
        r = requests.post(url, json=payload, headers=self.headers)
        r.raise_for_status()

        raw = msgpack.unpackb(r.content, raw=False)
        results = []
        for item in raw:
            # item layout: [score, id, meta_str, filter_str, ?, []]
            results.append({
                "score":  item[0] if len(item) > 0 else 0.0,
                "id":     item[1] if len(item) > 1 else "",
                "meta":   item[2] if len(item) > 2 else "",
                "filter": item[3] if len(item) > 3 else "",
            })
        return results

    # ------------------------------------------------------------------ health
    def health(self) -> dict:
        r = requests.get(self._url("/api/v1/health"), headers=self.headers)
        r.raise_for_status()
        return r.json()
