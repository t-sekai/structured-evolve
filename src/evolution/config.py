"""Local configuration loading for evolution experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tomllib


DEFAULT_API_CONFIG_PATH = Path("config/api_keys.toml")


def load_api_config(path: Path = DEFAULT_API_CONFIG_PATH) -> dict[str, Any]:
    """Load local API config from TOML, returning an empty config if absent."""
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def config_value(config: dict[str, Any], section: str, key: str) -> Any:
    """Return a non-empty config value, or None."""
    value = config.get(section, {}).get(key)
    if value == "":
        return None
    return value
