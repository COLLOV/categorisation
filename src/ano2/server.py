from __future__ import annotations

import os
from dotenv import load_dotenv

from .api import create_app
from .config import PipelineConfig


load_dotenv(override=False)
cfg_path = os.getenv("PIPELINE_CONFIG", "config/pipeline.example.yaml")
cfg = PipelineConfig.from_yaml(cfg_path)
app = create_app(cfg)

