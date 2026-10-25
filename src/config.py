"""Load project configuration and resolve paths safely."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# Load local environment variables only if a .env file exists.
load_dotenv(PROJECT_ROOT / ".env")


def load_config() -> dict[str, Any]:
    """Read the YAML configuration file."""
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {CONFIG_PATH}")
    return config


def resolve_data_path(config: dict[str, Any], override: str | None = None) -> Path:
    """Resolve the dataset path from CLI override, environment, or YAML config."""
    configured_path = (
        override
        or os.getenv("FRAUDSHIELD_DATA_PATH")
        or config.get("data", {}).get("raw_path")
    )
    if not configured_path:
        raise ValueError("No dataset path configured. Set data.raw_path in configs/config.yaml.")

    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()
