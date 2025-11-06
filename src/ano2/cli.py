from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

from .config import PipelineConfig
from .pipeline import Pipeline
from .log import get_logger


logger = get_logger(__name__)
app = typer.Typer(add_completion=False, help="ANO2 feedback categorization pipeline")


@app.command()
def run(config: str = typer.Option(..., "--config", "-c", help="Path to pipeline YAML")) -> None:
    load_dotenv(override=False)
    cfg = PipelineConfig.from_yaml(config)
    logger.info("Starting pipeline with config: %s", Path(config).resolve())
    p = Pipeline(cfg)
    p.run()
    logger.info("Done")
