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
        # Strict v2 prompt: enforce exact JSON and 3-way sentiment
        system_msg = (
            "Return only a single JSON object. No markdown, no backticks, no explanations. "
            "Your output is parsed by json.loads; any extra text will cause a failure."
        )
        instr = (
            "You are a precise classifier. Categorize feedback into: category, subcategory, sentiment.\n"
            "Output format (exactly): {\"category\":\"...\",\"subcategory\":\"...\",\"sentiment\":\"positive|neutral|negative\"}.\n"
            "Rules:\n"
            "- Sentiment MUST be one of: positive, neutral, negative.\n"
            "- If mixed/uncertain tone, choose 'neutral'.\n"
            "- Use short, human-readable labels for category/subcategory (max 3 words)."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": instr + "\nFeedback: L'app marche mais parfois elle rame un peu."},
            {"role": "assistant", "content": "{\"category\":\"Performance\",\"subcategory\":\"Intermittent slowdowns\",\"sentiment\":\"neutral\"}"},
            {"role": "user", "content": "Feedback: Très satisfait, aucun problème rencontré."},
            {"role": "assistant", "content": "{\"category\":\"Satisfaction\",\"subcategory\":\"No issues\",\"sentiment\":\"positive\"}"},
            {"role": "user", "content": f"Feedback: {text}"},
        ]

        body = {
            "model": self.model,
            "messages": messages,
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

        # Some local models wrap JSON in a fenced code block (```json ... ```)
        def _unwrap_fence(s: str) -> str:
            t = s.strip()
            if t.startswith("```") and t.endswith("```"):
                t = t[3:-3].strip()
                # Optional language tag like ```json
                if t.lower().startswith("json"):
                    t = t[4:].strip()
            return t

        raw = _unwrap_fence(content)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError(f"Model did not return valid JSON: {content[:200]}")
        # Strict validation
        for k in ("category", "subcategory", "sentiment"):
            if k not in parsed or not isinstance(parsed[k], str) or not parsed[k].strip():
                raise ValueError(f"Invalid field '{k}' in model output: {parsed}")
        sent = parsed["sentiment"].strip().lower()
        if sent not in ("positive", "neutral", "negative"):
            raise ValueError("Invalid sentiment '{0}'. Must be 'positive', 'neutral', or 'negative'.".format(parsed["sentiment"]))
        return {
            "category": parsed["category"].strip(),
            "subcategory": parsed["subcategory"].strip(),
            "sentiment": sent,
        }
