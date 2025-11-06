from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict

import httpx

from .log import get_logger


logger = get_logger(__name__)


def _required_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


@dataclass
class LLMClient:
    base_url: str
    api_key: str | None
    model: str

    @classmethod
    def from_env(cls) -> "LLMClient":
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = _required_env("LLM_MODEL")
        mode = os.getenv("LLM_MODE", "api")
        api_key = os.getenv("OPENAI_API_KEY")
        if mode == "api" and not api_key:
            raise RuntimeError("LLM_MODE=api requires OPENAI_API_KEY")
        # For local vLLM, api_key may be None; many servers accept empty keys.
        return cls(base_url=base_url, api_key=api_key, model=model)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def categorize(self, text: str) -> Dict[str, str]:
        prompt = (
            "You are a precise classifier. Categorize the single feedback into a top-level 'category', a more specific 'subcategory', and 'sentiment'.\n"
            "Rules:\n"
            "- Return STRICT JSON with keys: category, subcategory, sentiment.\n"
            "- Sentiment MUST be one of: positive, negative. No neutral.\n"
            "- Use concise, human-readable labels (max 3 words each).\n\n"
            f"Feedback: {text}"
        )
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return only JSON. No prose."},
                {"role": "user", "content": prompt},
            ],
            # Avoid provider-specific response_format to keep compatibility
            "temperature": 0,
        }
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        logger.debug("POST %s", url)
        with httpx.Client(timeout=60) as client:
            resp = client.post(url, headers=self._headers(), json=body)
        if resp.status_code >= 300:
            raise RuntimeError(f"LLM error {resp.status_code}: {resp.text}")
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(f"Model did not return valid JSON: {content[:200]}")
        # Strict validation
        for k in ("category", "subcategory", "sentiment"):
            if k not in parsed or not isinstance(parsed[k], str) or not parsed[k].strip():
                raise ValueError(f"Invalid field '{k}' in model output: {parsed}")
        sent = parsed["sentiment"].strip().lower()
        if sent not in ("positive", "negative"):
            raise ValueError(f"Invalid sentiment '{parsed['sentiment']}'. Must be 'positive' or 'negative'.")
        return {
            "category": parsed["category"].strip(),
            "subcategory": parsed["subcategory"].strip(),
            "sentiment": sent,
        }

