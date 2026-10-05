"""Optional, scalar-only Weights & Biases integration.

The package is imported lazily so configurations with ``wandb_enabled: false``
retain the original dependency-free behaviour.  No images, masks, prediction
volumes, model artifacts, or source code are logged by this module.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any
import importlib


def init_run(config: Any, dest: Path, *, wandb_module: Any | None = None) -> Any | None:
    """Start a scalar-only W&B run when explicitly enabled in ``config``."""
    if not config.wandb_enabled:
        return None

    if wandb_module is None:
        try:
            wandb_module = importlib.import_module("wandb")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "W&B logging is enabled, but the 'wandb' package is unavailable. "
                "Install project requirements before submitting the run."
            ) from exc

    kwargs: dict[str, Any] = {
        "project": config.wandb_project,
        "name": config.wandb_run_name,
        "tags": config.wandb_tags or [],
        "config": asdict(config),
        "dir": str(dest),
        "save_code": False,
    }
    if config.wandb_entity:
        kwargs["entity"] = config.wandb_entity

    return wandb_module.init(**kwargs)


def log_epoch(run: Any | None, metrics: dict[str, float | int]) -> None:
    """Log scalar epoch metrics; a disabled run is a no-op."""
    if run is not None:
        run.log(metrics)


def finish_run(run: Any | None) -> None:
    """Close an enabled run after normal training/early stopping completion."""
    if run is not None:
        run.finish()
