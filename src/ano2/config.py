from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class IOConfig(BaseModel):
    type: str = Field(default="csv")
    path: str
    text_field: str
    id_field: Optional[str] = None
    output_path: Optional[str] = None


class LLMConfig(BaseModel):
    mode: Optional[str] = Field(default=None, description="api | local. Defaults to env LLM_MODE")
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key_env: str = Field(default="OPENAI_API_KEY")


class EmbeddingConfig(BaseModel):
    provider: str = Field(default="local", description="local | openai")
    local_model: str = Field(default="paraphrase-multilingual-MiniLM-L12-v2")
    openai_model: Optional[str] = None


class ClusteringConfig(BaseModel):
    category_threshold: float = 0.2  # cosine distance
    subcategory_threshold: float = 0.2

    @field_validator("category_threshold", "subcategory_threshold")
    @classmethod
    def _validate_threshold(cls, v: float) -> float:
        if not (0.0 <= v <= 2.0):
            raise ValueError("threshold must be within [0, 2] cosine distance range")
        return v


class FieldNames(BaseModel):
    category: str = Field(default="Category")
    subcategory: str = Field(default="Sub Category")
    sentiment: str = Field(default="Sentiment")


class PipelineConfig(BaseModel):
    io: IOConfig
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    batch_size: int = 1
    field_names: FieldNames = Field(default_factory=FieldNames)
    limit: int | None = Field(default=None, description="Process only the first N rows")

    @staticmethod
    def from_yaml(path: str | Path) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return PipelineConfig(**raw)
