"""Hardware definitions — one file per hardware target."""

from __future__ import annotations

from opcompass.registry import discover_hardware, get_hardware


def list_hardware() -> dict[str, str]:
    """Return {name: description} for all registered hardware targets."""
    return {name: f"{cls.vendor} {cls.name}" for name, cls in discover_hardware().items()}


__all__ = ["discover_hardware", "get_hardware", "list_hardware"]
