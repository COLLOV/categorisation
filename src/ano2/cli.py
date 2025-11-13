from __future__ import annotations

from pathlib import Path

import typer
from typer import Option
from dotenv import load_dotenv

from .config import PipelineConfig
from .pipeline import Pipeline
from .log import get_logger


logger = get_logger(__name__)
app = typer.Typer(add_completion=False, help="ANO2 feedback categorization pipeline")
app.pretty_exceptions_enable = False
app.pretty_exceptions_short = True


@app.callback(invoke_without_command=True)
def _default(
    ctx: typer.Context,
    config: str | None = Option(None, "--config", "-c", help="Path to pipeline YAML"),
) -> None:
    """Allow running without subcommand: `ano2 -c config.yaml`.

    If a subcommand is provided, this callback just returns and lets it run.
    """
    if ctx.invoked_subcommand is not None:
        return
    if not config:
        typer.echo("Usage: ano2 run -c <config.yaml>  or  ano2 -c <config.yaml>", err=True)
        raise typer.Exit(code=2)
    load_dotenv(override=False)
    cfg = PipelineConfig.from_yaml(config)
    logger.info("Starting pipeline with config: %s", Path(config).resolve())
    p = Pipeline(cfg)
    p.run()
    logger.info("Done")
    raise typer.Exit(code=0)


@app.command()
def run(config: str = typer.Option(..., "--config", "-c", help="Path to pipeline YAML")) -> None:
    load_dotenv(override=False)
    cfg = PipelineConfig.from_yaml(config)
    logger.info("Starting pipeline with config: %s", Path(config).resolve())
    p = Pipeline(cfg)
    p.run()
    logger.info("Done")
