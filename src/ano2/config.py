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
    # File outputs (choose either output_path or output_dir)
    output_path: Optional[str] = None
    output_dir: Optional[str] = None
    output_basename: str = Field(default="categorized.csv")

    add_timestamp_column: bool = Field(default=False)
    timestamp_column_name: str = Field(default="processed_at")
    # Timestamp formatting
    append_timestamp_to_output_path: bool = Field(default=False)
    timestamp_subdir: bool = Field(default=False)
    timestamp_format: str = Field(default="%Y%m%d-%H%M%S")
    # Summary outputs
    write_summary: bool = Field(default=False, description="Write counts recap to a JSON file")
    summary_path: Optional[str] = Field(default=None, description="Optional explicit path for summary JSON")
    summary_basename: Optional[str] = Field(default=None, description="Optional JSON filename when using output_dir (defaults to <stem>_summary.json)")


class LLMConfig(BaseModel):
    mode: Optional[str] = Field(default=None, description="api | local. Defaults to env LLM_MODE")
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key_env: str = Field(default="OPENAI_API_KEY")
    # Response enforcement and robustness
    strict_json: bool = Field(default=True, description="Require model to return a valid JSON object")
    json_mode: bool = Field(default=False, description="If provider supports it, set response_format=json_object")
    http_timeout: float = Field(default=60.0, description="HTTP timeout in seconds")
    max_tokens: Optional[int] = Field(default=None, description="Optional max tokens for the response")
    max_retries: int = Field(default=1, description="Retries on invalid JSON (strict only)")
    retry_invalid_json: bool = Field(default=True, description="Retry once with corrective hint if invalid JSON")


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
    keywords: str = Field(default="Keywords")
    # Optional plain-text rendering of keywords (joined string) for NL SQL
    keywords_text: Optional[str] = Field(default=None)


class KeywordsConfig(BaseModel):
    single_words_only: bool = Field(default=True, description="Only single words, no phrases/compounds")
    enforce_in_text: bool = Field(default=True, description="Each keyword must appear in the text")
    drop_generic: bool = Field(default=True, description="Drop generic/filler terms")
    min_length: int = Field(default=2, description="Minimum keyword length after normalization")


class PipelineConfig(BaseModel):
    io: IOConfig
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    batch_size: int = 1
    field_names: FieldNames = Field(default_factory=FieldNames)
    limit: int | None = Field(default=None, description="Process only the first N rows")
    workers: int = Field(default=1, description="Parallel workers for LLM calls (>=1)")
    keywords: KeywordsConfig = Field(default_factory=KeywordsConfig)

    @staticmethod
    def from_yaml(path: str | Path) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return PipelineConfig(**raw)
