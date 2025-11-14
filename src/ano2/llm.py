from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict
import re
import unicodedata

import httpx

from .log import get_logger
from .config import LLMConfig, KeywordsConfig


logger = get_logger(__name__)


def _required_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


def _unwrap_fence(s: str) -> str:
    t = s.strip()
    if t.startswith("```") and t.endswith("```"):
        t = t[3:-3].strip()
        # Optional language tag like ```json
        tl = t.lower()
        if tl.startswith("json"):
            t = t[4:].strip()
    return t


@dataclass
class LLMClient:
    base_url: str
    api_key: str | None
    model: str
    # Runtime behavior
    strict_json: bool = True
    json_mode: bool = False
    http_timeout: float = 60.0
    max_tokens: int | None = None
    max_retries: int = 1
    retry_invalid_json: bool = True
    # Keywords rules
    kw_single_words_only: bool = True
    kw_enforce_in_text: bool = True
    kw_drop_generic: bool = True
    kw_min_length: int = 2

    @classmethod
    def from_env(cls) -> "LLMClient":
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = _required_env("LLM_MODEL")
        mode = os.getenv("LLM_MODE", "api")
        api_key = os.getenv("OPENAI_API_KEY")
        if mode == "api" and not api_key:
            raise RuntimeError("LLM_MODE=api requires OPENAI_API_KEY")
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            strict_json=os.getenv("LLM_STRICT_JSON", "1") not in ("0", "false", "False"),
            json_mode=os.getenv("LLM_JSON_MODE", "0").lower() not in ("0", "false", ""),
            http_timeout=float(os.getenv("LLM_HTTP_TIMEOUT", "60")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "0")) or None,
            max_retries=max(0, int(os.getenv("LLM_MAX_RETRIES", "1"))),
            retry_invalid_json=os.getenv("LLM_RETRY_INVALID_JSON", "1").lower() not in ("0", "false", ""),
            kw_single_words_only=os.getenv("KEYWORDS_SINGLE_WORDS_ONLY", "1") not in ("0", "false", "False"),
            kw_enforce_in_text=os.getenv("KEYWORDS_ENFORCE_IN_TEXT", "1") not in ("0", "false", "False"),
            kw_drop_generic=os.getenv("KEYWORDS_DROP_GENERIC", "1") not in ("0", "false", "False"),
            kw_min_length=max(1, int(os.getenv("KEYWORDS_MIN_LENGTH", "2"))),
        )

    @classmethod
    def from_config(cls, llm: LLMConfig, keywords: KeywordsConfig | None = None) -> "LLMClient":
        # Prefer YAML, fall back to env
        base_url = llm.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = llm.model or _required_env("LLM_MODEL")
        mode = llm.mode or os.getenv("LLM_MODE", "api")
        api_key = os.getenv(llm.api_key_env)
        if mode == "api" and not api_key:
            raise RuntimeError("LLM_MODE=api requires OPENAI_API_KEY (or configured api_key_env)")
        kw = keywords or KeywordsConfig()
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            strict_json=llm.strict_json,
            json_mode=llm.json_mode,
            http_timeout=float(llm.http_timeout),
            max_tokens=llm.max_tokens,
            max_retries=max(0, int(llm.max_retries)),
            retry_invalid_json=bool(llm.retry_invalid_json),
            kw_single_words_only=kw.single_words_only,
            kw_enforce_in_text=kw.enforce_in_text,
            kw_drop_generic=kw.drop_generic,
            kw_min_length=max(1, int(kw.min_length)),
        )

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def categorize(self, text: str) -> Dict[str, str | list[str]]:
        # Strict v2 prompt: enforce exact JSON and 3-way sentiment
        system_msg = (
            "Return only a single JSON object. No markdown, no backticks, no explanations. "
            "Your output is parsed by json.loads; any extra text will cause a failure."
        )
        instr = (
            "You are a precise classifier. Categorize feedback and extract keywords.\n"
            'Output format (exactly): {"category":"...","subcategory":"...","sentiment":"positive|neutral|negative","keywords":["k1","k2","k3"]}.\n'
            "Rules:\n"
            "- Sentiment MUST be one of: positive, neutral, negative.\n"
            "- If mixed/uncertain tone, choose 'neutral'.\n"
            "- Use short, human-readable labels for category/subcategory (max 3 words).\n"
            "- keywords: 3 to 8 single words (no phrases, no hyphenated compounds), each word must appear verbatim in the feedback as a standalone word; keep the text language; lowercase; no emojis; no duplicates; avoid generic fillers (e.g., 'issue', 'problem', 'application', 'user', 'support'). If few are available, return fewer."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": instr + "\nFeedback: L'app marche mais parfois elle rame un peu."},
            {"role": "assistant", "content": '{"category":"Performance","subcategory":"Intermittent slowdowns","sentiment":"neutral","keywords":["rame"]}'},
            {"role": "user", "content": "Feedback: Très satisfait, aucun problème rencontré."},
            {"role": "assistant", "content": '{"category":"Satisfaction","subcategory":"No issues","sentiment":"positive","keywords":["satisfait"]}'},
            {"role": "user", "content": f"Feedback: {text}"},
        ]

        # Prepare request body and runtime options
        base_body = {
            "model": self.model,
            "messages": None,  # filled per attempt
            "temperature": 0,
        }
        # Optional max tokens
        if self.max_tokens and self.max_tokens > 0:
            base_body["max_tokens"] = self.max_tokens
        # Optional JSON mode for providers that support it (e.g., OpenAI)
        if self.json_mode:
            base_body["response_format"] = {"type": "json_object"}

        http_timeout = float(self.http_timeout)

        # Retries on invalid JSON (strict mode only)
        max_retries = int(self.max_retries)
        allow_retry_invalid = bool(self.retry_invalid_json)

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        logger.debug("POST %s", url)

        # Attempt loop (helps recover occasional malformed outputs)
        attempt_messages = list(messages)
        content = ""
        for attempt in range(max_retries + 1):
            body = dict(base_body)
            body["messages"] = attempt_messages
            with httpx.Client(timeout=http_timeout) as client:
                resp = client.post(url, headers=self._headers(), json=body)
            if resp.status_code >= 300:
                raise RuntimeError(f"LLM error {resp.status_code}: {resp.text}")
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()

            raw = _unwrap_fence(content)
            strict_json = bool(self.strict_json)
            parsed = None
            if strict_json:
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as e:
                    # Optionally retry with a corrective instruction
                    if allow_retry_invalid and attempt < max_retries:
                        logger.warning("Invalid JSON on attempt %d: %s... Retrying once.", attempt + 1, content[:120])
                        attempt_messages = attempt_messages + [
                            {
                                "role": "user",
                                "content": (
                                    "Your previous answer was not valid JSON. Return ONLY one JSON object with keys "
                                    "category (string), subcategory (string), sentiment ('positive'|'neutral'|'negative'), "
                                    "keywords (array of single-word strings). No markdown, no extra text."
                                ),
                            }
                        ]
                        continue
                    else:
                        raise ValueError(f"Model did not return valid JSON (strict): {content[:200]}") from e
            else:
                # Lenient parsing for non-conformant providers (dev/local only)
                last_err: Exception | None = None
                # Attempt 1: direct JSON object/array
                try:
                    parsed = json.loads(raw)
                except Exception as e:
                    last_err = e
                # Attempt 2: content may be a JSON-escaped string of the object (wrapped quotes)
                if parsed is None and raw.strip().startswith("\"") and raw.strip().endswith("\""):
                    try:
                        inner = json.loads(raw.strip())
                        if isinstance(inner, str):
                            parsed = json.loads(inner)
                            raw = inner
                    except Exception as e:
                        last_err = e
                # Attempt 3: escaped JSON object without outer quotes, e.g. {\"a\":1}
                if parsed is None and raw.strip().startswith("{\\\"") and '\\"' in raw:
                    try:
                        unescaped = json.loads("\"" + raw + "\"")  # decode escapes
                        parsed = json.loads(unescaped)
                        raw = unescaped
                    except Exception as e:
                        last_err = e
                # Attempt 4: extract the first top-level {...} window if extra text surrounds JSON
                if parsed is None and "{" in raw and "}" in raw:
                    i = raw.find("{")
                    j = raw.rfind("}")
                    if j > i >= 0:
                        candidate = raw[i : j + 1]
                        try:
                            parsed = json.loads(candidate)
                            raw = candidate
                        except Exception as e:
                            last_err = e
                if parsed is None:
                    if allow_retry_invalid and attempt < max_retries:
                        logger.warning("Lenient parse failed on attempt %d: %s... Retrying once.", attempt + 1, content[:120])
                        attempt_messages = attempt_messages + [
                            {
                                "role": "user",
                                "content": (
                                    "Your previous answer was not valid JSON. Return ONLY one JSON object with keys "
                                    "category, subcategory, sentiment, keywords (array of single words)."
                                ),
                            }
                        ]
                        continue
                    raise ValueError(f"Model did not return valid JSON: {content[:200]}")

            # From here we have parsed
            break

        # (helper _unwrap_fence is defined at module scope)

        # 'parsed' is guaranteed to be set here
        # Strict validation
        for k in ("category", "subcategory", "sentiment"):
            if k not in parsed or not isinstance(parsed[k], str) or not parsed[k].strip():
                raise ValueError(f"Invalid field '{k}' in model output: {parsed}")
        sent = parsed["sentiment"].strip().lower()
        if sent not in ("positive", "neutral", "negative"):
            raise ValueError("Invalid sentiment '{0}'. Must be 'positive', 'neutral', or 'negative'.".format(parsed["sentiment"]))
        # keywords: ensure list[str]
        kws: list[str]
        if "keywords" not in parsed:
            raise ValueError(f"Missing 'keywords' in model output: {parsed}")
        if isinstance(parsed["keywords"], list):
            kws = [str(x).strip() for x in parsed["keywords"] if str(x).strip()]
        elif isinstance(parsed["keywords"], str):
            # allow comma-separated string as a fallback
            kws = [x.strip() for x in parsed["keywords"].split(",") if x.strip()]
        else:
            raise ValueError(f"Invalid 'keywords' type: {type(parsed['keywords'])}")
        # Deduplicate while preserving order, lowercase
        seen = set()
        clean_kws: list[str] = []
        for k in kws:
            kl = k.lower()
            if kl not in seen:
                seen.add(kl)
                clean_kws.append(kl)
        
        # Optional: enforce that keywords are grounded in the input text and non-generic
        def _strip_accents(s: str) -> str:
            return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

        def _norm(s: str) -> str:
            t = _strip_accents(s.lower())
            # Normalize punctuation to spaces
            t = re.sub(r"[\s\-_/]+", " ", t)
            t = re.sub(r"[^\w\s]", " ", t)
            t = re.sub(r"\s+", " ", t).strip()
            return t

        def _contains_word(text_norm: str, word_norm: str) -> bool:
            if not word_norm:
                return False
            return re.search(rf"\b{re.escape(word_norm)}\b", text_norm) is not None

        enforce = bool(self.kw_enforce_in_text)
        drop_generic = bool(self.kw_drop_generic)
        single_words_only = bool(self.kw_single_words_only)
        min_len = int(self.kw_min_length)

        text_norm = _norm(text)
        # Build a map from normalized token -> original lowercased token as it appears in text
        norm_to_orig: Dict[str, str] = {}
        for m in re.finditer(r"\w+", text, flags=re.UNICODE):
            orig_tok = m.group(0)
            norm_tok = _strip_accents(orig_tok.lower())
            if norm_tok and norm_tok not in norm_to_orig:
                norm_to_orig[norm_tok] = orig_tok.lower()
        generic_singletons = {
            # English
            "issue", "issues", "problem", "problems", "user", "users", "client", "clients",
            "app", "application", "service", "services", "support", "team", "software", "system",
            # French
            "problème", "probleme", "problèmes", "problemes", "utilisateur", "utilisateurs",
            "client", "clients", "appli", "application", "service", "services", "support", "équipe", "equipe",
        }
        http_code_whitelist = {"400","401","403","404","408","409","410","422","429","500","501","502","503","504"}

        def _is_generic(kw_norm: str) -> bool:
            if len(kw_norm) < min_len:
                return True
            if kw_norm.isdigit() and kw_norm not in http_code_whitelist:
                return True
            if kw_norm in generic_singletons:
                return True
            return False

        # Build candidate terms (single words only when enforced)
        candidates: list[str] = []
        for kw in clean_kws:
            kn = _norm(kw)
            if single_words_only and (" " in kn):
                # Drop compound/hyphenated/multi-word suggestions entirely
                continue
            candidates.append(kn)

        # Filter candidates
        filtered: list[str] = []
        seen_terms = set()
        for term in candidates:
            if term in seen_terms:
                continue
            seen_terms.add(term)
            if drop_generic and _is_generic(term):
                continue
            if enforce and not _contains_word(text_norm, term):
                continue
            filtered.append(term)

        # If everything got filtered and enforcement is on, allow returning 0..n keywords (strict grounding).
        # Map normalized terms back to original lowercase tokens when available
        final_kws = [norm_to_orig.get(t, t) for t in filtered] if filtered or enforce else clean_kws
        return {
            "category": parsed["category"].strip(),
            "subcategory": parsed["subcategory"].strip(),
            "sentiment": sent,
            "keywords": final_kws,
        }
