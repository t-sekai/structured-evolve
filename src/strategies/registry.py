"""Strategy lookup helpers."""

from __future__ import annotations

from src.strategies.fixed import FixedScheduleStrategy
from src.strategies.metaschedule import MetaScheduleStrategy


_STRATEGIES = {
    FixedScheduleStrategy.name: FixedScheduleStrategy,
    MetaScheduleStrategy.name: MetaScheduleStrategy,
}


def available_strategy_names() -> tuple[str, ...]:
    """Return stable CLI names for all available strategies."""
    return tuple(_STRATEGIES)


def get_strategy(name: str):
    """Create a strategy by CLI name."""
    try:
        return _STRATEGIES[name]()
    except KeyError as err:
        choices = ", ".join(available_strategy_names())
        raise ValueError(f"Unknown strategy '{name}'. Available strategies: {choices}") from err
