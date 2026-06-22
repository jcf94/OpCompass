"""Auto-discovery registry for operators and hardware definitions.

Scans the `operators/` and `hardware/` sub-packages for classes that inherit
from `Operator` / `Hardware`.  Adding a new operator or hardware is as simple
as dropping a new `.py` file into the right directory — no manual
registration needed.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opcompass.operators.base import Operator
    from opcompass.hardware.base import Hardware


def _discover(package_name: str, base_class: type) -> dict[str, type]:
    """Walk *package_name* and return a {name: subclass} dict for all
    concrete subclasses of *base_class* found."""
    found: dict[str, type] = {}

    try:
        package = importlib.import_module(package_name)
    except ImportError:
        return found

    for _, mod_name, _ in pkgutil.walk_packages(
        package.__path__, prefix=package_name + "."
    ):
        if mod_name.endswith(".__init__"):
            continue
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue

        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, base_class)
                and attr is not base_class
            ):
                # Use the class's `name` attribute as key (every Operator / Hardware has one).
                # Skip intermediate base classes (e.g. NvidiaAmpere) that
                # intentionally leave `name` empty to avoid registration.
                key: str = getattr(attr, "name", "")
                if not key:
                    continue
                found[key] = attr

    return found


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_operators() -> dict[str, type[Operator]]:
    """Return all registered operators as {name: OperatorClass}."""
    from opcompass.operators.base import Operator

    return _discover("opcompass.operators", Operator)


def discover_hardware() -> dict[str, type[Hardware]]:
    """Return all registered hardware targets as {name: HardwareClass}."""
    from opcompass.hardware.base import Hardware

    return _discover("opcompass.hardware", Hardware)


def get_operator(name: str) -> type[Operator]:
    """Look up a single operator by name.  Raises KeyError if not found."""
    ops = discover_operators()
    if name not in ops:
        raise KeyError(f"Unknown operator '{name}'. Available: {list(ops.keys())}")
    return ops[name]


def get_hardware(name: str) -> type[Hardware]:
    """Look up a single hardware target by name.  Raises KeyError if not found."""
    hw = discover_hardware()
    if name not in hw:
        raise KeyError(f"Unknown hardware '{name}'. Available: {list(hw.keys())}")
    return hw[name]
