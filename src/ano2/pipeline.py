from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from .config import PipelineConfig
from .embed import Embeddings
from .llm import LLMClient
from .log import get_logger


logger = get_logger(__name__)


def _normalize_label(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip("-_")
    return s


def _cluster_labels(labels: List[str], embeddings: Embeddings, threshold: float) -> List[int]:
    if len(labels) == 0:
        return []
    if len(labels) == 1:
        return [0]
    embs = embeddings.encode(labels)
    # Agglomerative with cosine distance via precomputed metric is messy; we can use euclidean on normalized embeddings
    # because cosine distance on normalized vectors is proportional to euclidean distance.
    # Convert cosine distance threshold to euclidean threshold: d_eucl^2 = 2*(1-cos)
    cos_thr = threshold
    eucl_thr = np.sqrt(max(0.0, 2.0 * (1.0 - (1.0 - cos_thr))))  # 1 - cos = cos_thr -> eucl^2 = 2*cos_thr
    eucl_thr = np.sqrt(2.0 * cos_thr)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=eucl_thr,
        metric="euclidean",
        linkage="average",
    )
    labels_arr = clustering.fit_predict(embs)
    return list(map(int, labels_arr))


def _canonical_labels(groups: Dict[int, List[str]]) -> Dict[int, str]:
    canon = {}
    for gid, items in groups.items():
        norm = [_normalize_label(x) for x in items]
        # choose the most frequent normalized label, tie-breaker shortest
        freq = Counter(norm)
        best_norm, _ = max(freq.items(), key=lambda kv: (kv[1], -len(kv[0])))
        # pick an original variant matching best_norm with shortest length
        candidates = [x for x in items if _normalize_label(x) == best_norm]
        best = min(candidates, key=len)
        canon[gid] = best
    return canon


@dataclass
class Pipeline:
    cfg: PipelineConfig

    def run(self) -> pd.DataFrame:
        df = self._load()
        logger.info("Loaded %d rows", len(df))
        llm = LLMClient.from_env()
        emb = Embeddings.from_config(
            provider=self.cfg.embedding.provider,
            local_model=self.cfg.embedding.local_model,
            openai_model=self.cfg.embedding.openai_model,
        )

        results = []
        for i, row in df.iterrows():
            text = str(row[self.cfg.io.text_field])
            parsed = llm.categorize(text)
            results.append(parsed)
            if (i + 1) % 20 == 0:
                logger.info("Processed %d/%d", i + 1, len(df))

        df_pred = pd.DataFrame(results)
        df_combined = pd.concat([df.reset_index(drop=True), df_pred], axis=1)

        # Canonicalize categories
        cat_labels = list(df_combined["category"].astype(str))
        cat_clusters = _cluster_labels(cat_labels, emb, self.cfg.clustering.category_threshold)
        cat_groups: Dict[int, List[str]] = defaultdict(list)
        for label, gid in zip(cat_labels, cat_clusters):
            cat_groups[gid].append(label)
        cat_canon = _canonical_labels(cat_groups)
        df_combined["category_canon"] = [cat_canon[g] for g in cat_clusters]

        # Canonicalize subcategories within each canonical category
        subcanon_out = []
        for canon_cat in df_combined["category_canon"].unique():
            mask = df_combined["category_canon"] == canon_cat
            sub_labels = list(df_combined.loc[mask, "subcategory"].astype(str))
            sub_clusters = _cluster_labels(sub_labels, emb, self.cfg.clustering.subcategory_threshold)
            sub_groups: Dict[int, List[str]] = defaultdict(list)
            for label, gid in zip(sub_labels, sub_clusters):
                sub_groups[gid].append(label)
            sub_canon = _canonical_labels(sub_groups)
            # write back
            idxs = df_combined.index[mask].tolist()
            for idx, gid in zip(idxs, sub_clusters):
                subcanon_out.append((idx, sub_canon[gid]))
        sub_map = {idx: label for idx, label in subcanon_out}
        df_combined["subcategory_canon"] = df_combined.index.map(sub_map.get)

        # Rename to configured field names
        out = df_combined.copy()
        out.rename(
            columns={
                "category_canon": self.cfg.field_names.category,
                "subcategory_canon": self.cfg.field_names.subcategory,
                "sentiment": self.cfg.field_names.sentiment,
            },
            inplace=True,
        )
        # Drop intermediate columns
        out.drop(columns=["category", "subcategory"], inplace=True)

        if self.cfg.io.output_path:
            out_path = self.cfg.io.output_path
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            out.to_csv(out_path, index=False)
            logger.info("Saved output to %s", out_path)
        return out

    def _load(self) -> pd.DataFrame:
        io = self.cfg.io
        if io.type == "csv":
            df = pd.read_csv(io.path)
        elif io.type == "jsonl":
            df = pd.read_json(io.path, lines=True)
        else:
            raise ValueError(f"Unsupported io.type: {io.type}")
        # Validate field presence
        if io.text_field not in df.columns:
            raise KeyError(f"text_field '{io.text_field}' not in columns: {list(df.columns)}")
        return df

