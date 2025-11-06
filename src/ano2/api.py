from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from .config import PipelineConfig
from .llm import LLMClient
from .embed import Embeddings
from .log import get_logger


logger = get_logger(__name__)


class Item(BaseModel):
    id: Optional[str] = None
    text: str


class ItemsRequest(BaseModel):
    items: List[Item]


def create_app(cfg: PipelineConfig) -> FastAPI:
    app = FastAPI(title="ANO2 Categorization API")
    load_dotenv(override=False)
    llm = LLMClient.from_env()
    emb = Embeddings.from_config(
        provider=cfg.embedding.provider,
        local_model=cfg.embedding.local_model,
        openai_model=cfg.embedding.openai_model,
    )

    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/categorize")
    def categorize(req: ItemsRequest) -> Dict[str, Any]:
        try:
            rows: List[Dict[str, Any]] = []
            for idx, item in enumerate(req.items):
                out = llm.categorize(item.text)
                rows.append({
                    "id": item.id if item.id is not None else str(idx),
                    "text": item.text,
                    **out,
                })
            # Canonicalize categories across the batch
            from .pipeline import _cluster_labels, _canonical_labels

            cats = [r["category"] for r in rows]
            cids = _cluster_labels(cats, emb, cfg.clustering.category_threshold)
            groups: Dict[int, List[str]] = {}
            for c, gid in zip(cats, cids):
                groups.setdefault(gid, []).append(c)
            cat_canon = _canonical_labels(groups)
            for r, gid in zip(rows, cids):
                r["category"] = cat_canon[gid]

            # subcategories within category
            by_cat: Dict[str, List[int]] = {}
            for i, r in enumerate(rows):
                by_cat.setdefault(r["category"], []).append(i)
            for cat, idxs in by_cat.items():
                subs = [rows[i]["subcategory"] for i in idxs]
                sids = _cluster_labels(subs, emb, cfg.clustering.subcategory_threshold)
                sgroups: Dict[int, List[str]] = {}
                for s, gid in zip(subs, sids):
                    sgroups.setdefault(gid, []).append(s)
                sub_canon = _canonical_labels(sgroups)
                for i, gid in zip(idxs, sids):
                    rows[i]["subcategory"] = sub_canon[gid]

            return {"items": rows}
        except Exception as e:
            logger.exception("categorize failed")
            raise HTTPException(status_code=400, detail=str(e))

    return app
