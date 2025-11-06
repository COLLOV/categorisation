from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import json
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from datetime import datetime, timezone
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def _canonical_labels_weighted(groups: Dict[int, List[str]], counts: Dict[str, int]) -> Dict[int, str]:
    canon = {}
    for gid, items in groups.items():
        # Sum counts by normalized form
        agg: Dict[str, int] = {}
        for x in items:
            nx = _normalize_label(x)
            agg[nx] = agg.get(nx, 0) + int(counts.get(x, 1))
        best_norm, _ = max(agg.items(), key=lambda kv: (kv[1], -len(kv[0])))
        # pick the shortest original matching best_norm
        candidates = [x for x in items if _normalize_label(x) == best_norm]
        best = min(candidates, key=len)
        canon[gid] = best
    return canon


@dataclass
class Pipeline:
    cfg: PipelineConfig

    def run(self) -> pd.DataFrame:
        df = self._load()
        if self.cfg.limit is not None and self.cfg.limit > 0:
            df = df.head(self.cfg.limit)
            logger.info("Loaded %d rows (limited to %d)", len(df), self.cfg.limit)
        else:
            logger.info("Loaded %d rows", len(df))
        llm = LLMClient.from_env()
        emb = Embeddings.from_config(
            provider=self.cfg.embedding.provider,
            local_model=self.cfg.embedding.local_model,
            openai_model=self.cfg.embedding.openai_model,
        )

        logger.info("Classifying %d rows (first call may take a few seconds)", len(df))
        texts = [str(df.iloc[i][self.cfg.io.text_field]) for i in range(len(df))]
        results: List[Dict[str, str]] = [None] * len(texts)  # type: ignore
        if self.cfg.workers <= 1:
            for i, text in enumerate(tqdm(texts, total=len(texts), desc="Classifying", unit="row")):
                results[i] = llm.categorize(text)
        else:
            with ThreadPoolExecutor(max_workers=self.cfg.workers) as ex, tqdm(total=len(texts), desc="Classifying", unit="row") as pbar:
                futs = {ex.submit(llm.categorize, t): i for i, t in enumerate(texts)}
                for fut in as_completed(futs):
                    i = futs[fut]
                    results[i] = fut.result()
                    pbar.update(1)

        df_pred = pd.DataFrame(results)
        df_combined = pd.concat([df.reset_index(drop=True), df_pred], axis=1)

        # Canonicalize categories (dedupe before clustering)
        cat_labels = list(df_combined["category"].astype(str))
        cat_counts = Counter(cat_labels)
        cat_uniques = list(cat_counts.keys())
        cat_u_clusters = _cluster_labels(cat_uniques, emb, self.cfg.clustering.category_threshold)
        cat_u_groups: Dict[int, List[str]] = defaultdict(list)
        for label, gid in zip(cat_uniques, cat_u_clusters):
            cat_u_groups[gid].append(label)
        cat_u_canon = _canonical_labels_weighted(cat_u_groups, cat_counts)
        # map each unique to canonical
        cat_map = {label: cat_u_canon[gid] for label, gid in zip(cat_uniques, cat_u_clusters)}
        df_combined["category_canon"] = [cat_map[l] for l in cat_labels]

        # Canonicalize subcategories within each canonical category
        subcanon_out = []
        for canon_cat in tqdm(df_combined["category_canon"].unique(), desc="Subcategory clustering", unit="cat"):
            mask = df_combined["category_canon"] == canon_cat
            sub_labels = list(df_combined.loc[mask, "subcategory"].astype(str))
            sub_counts = Counter(sub_labels)
            sub_uniques = list(sub_counts.keys())
            if len(sub_uniques) == 0:
                continue
            sub_u_clusters = _cluster_labels(sub_uniques, emb, self.cfg.clustering.subcategory_threshold)
            sub_u_groups: Dict[int, List[str]] = defaultdict(list)
            for label, gid in zip(sub_uniques, sub_u_clusters):
                sub_u_groups[gid].append(label)
            sub_u_canon = _canonical_labels_weighted(sub_u_groups, sub_counts)
            sub_map = {label: sub_u_canon[gid] for label, gid in zip(sub_uniques, sub_u_clusters)}
            # write back
            idxs = df_combined.index[mask].tolist()
            for idx, label in zip(idxs, sub_labels):
                subcanon_out.append((idx, sub_map[label]))
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

        if self.cfg.io.add_timestamp_column:
            col = self.cfg.io.timestamp_column_name
            if col in out.columns:
                raise ValueError(f"Timestamp column '{col}' already exists in output")
            ts_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            out[col] = ts_iso

        if self.cfg.io.output_path:
            base_path = self.cfg.io.output_path
            out_path = base_path
            if self.cfg.io.append_timestamp_to_output_path:
                ts_short = datetime.now(timezone.utc).strftime(self.cfg.io.timestamp_format)
                root, ext = os.path.splitext(base_path)
                out_path = f"{root}_{ts_short}{ext or '.csv'}"
            dirn = os.path.dirname(out_path) or "."
            os.makedirs(dirn, exist_ok=True)
            out.to_csv(out_path, index=False)
            logger.info("Saved output to %s", out_path)

        # Summary counts
        cat_col = self.cfg.field_names.category
        sub_col = self.cfg.field_names.subcategory
        cat_series = out[cat_col].value_counts().sort_values(ascending=False)
        logger.info("Category counts (%d unique):", cat_series.shape[0])
        for label, count in cat_series.items():
            logger.info("  %s: %d", label, int(count))

        sub_df = (
            out.groupby([cat_col, sub_col]).size().reset_index(name="count").sort_values([cat_col, "count"], ascending=[True, False])
        )
        logger.info("Subcategory counts by category:")
        curr = None
        for _, row in sub_df.iterrows():
            cat = row[cat_col]
            sub = row[sub_col]
            cnt = int(row["count"])
            if cat != curr:
                logger.info("  %s:", cat)
                curr = cat
            logger.info("    - %s: %d", sub, cnt)

        # Persist summary if configured
        if self.cfg.io.write_summary:
            if self.cfg.io.summary_path:
                base = self.cfg.io.summary_path
            elif self.cfg.io.output_path:
                root, _ = os.path.splitext(self.cfg.io.output_path)
                base = f"{root}_summary.json"
            else:
                raise ValueError("write_summary=true requires either io.summary_path or io.output_path to derive from")

            out_path = base
            if self.cfg.io.append_timestamp_to_output_path:
                ts_short = datetime.now(timezone.utc).strftime(self.cfg.io.timestamp_format)
                root, ext = os.path.splitext(base)
                out_path = f"{root}_{ts_short}{ext or '.json'}"
            dirn = os.path.dirname(out_path) or "."
            os.makedirs(dirn, exist_ok=True)

            summary = {
                "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "total_rows": int(out.shape[0]),
                "category_counts": [{"category": str(lbl), "count": int(cnt)} for lbl, cnt in cat_series.items()],
                "subcategory_counts": [
                    {"category": str(r[cat_col]), "subcategory": str(r[sub_col]), "count": int(r["count"])}
                    for _, r in sub_df.iterrows()
                ],
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            logger.info("Saved summary to %s", out_path)
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
