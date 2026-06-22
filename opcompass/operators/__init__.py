"""Operator definitions — one file per operator."""

from __future__ import annotations

from opcompass.registry import discover_operators, get_operator


def list_operators() -> dict[str, str]:
    """Return {name: description} for all registered operators."""
    return {name: cls.description for name, cls in discover_operators().items()}


__all__ = ["discover_operators", "get_operator", "list_operators"]
