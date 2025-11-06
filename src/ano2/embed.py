from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List

import httpx
import numpy as np
from sentence_transformers import SentenceTransformer

from .log import get_logger


logger = get_logger(__name__)


@dataclass
class Embeddings:
    provider: str = "local"  # local | openai
    local_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    openai_model: str | None = None

    _st_model: SentenceTransformer | None = None

    @classmethod
    def from_config(cls, provider: str, local_model: str, openai_model: str | None) -> "Embeddings":
        return cls(provider=provider, local_model=local_model, openai_model=openai_model)

    def _ensure_local(self) -> None:
        if self._st_model is None:
            logger.info("Loading embeddings model: %s", self.local_model)
            self._st_model = SentenceTransformer(self.local_model)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        if self.provider == "local":
            self._ensure_local()
            assert self._st_model is not None
            embs = self._st_model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
            return embs.astype(np.float32)
        elif self.provider == "openai":
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY required for openai embeddings provider")
            model = self.openai_model or "text-embedding-3-small"
            body = {"model": model, "input": list(texts)}
            with httpx.Client(timeout=60) as client:
                r = client.post(
                    f"{base_url}/embeddings",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=body,
                )
            if r.status_code >= 300:
                raise RuntimeError(f"Embeddings error {r.status_code}: {r.text}")
            data = r.json()
            vectors = [d["embedding"] for d in data["data"]]
            arr = np.array(vectors, dtype=np.float32)
            # Normalize to cosine space
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
            return (arr / norms).astype(np.float32)
        else:
            raise ValueError(f"Unknown embeddings provider: {self.provider}")

