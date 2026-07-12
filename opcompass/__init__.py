"""OpCompass — SOL theoretical peak performance estimator for GPU operators."""

import os
from pathlib import Path

__version__ = "0.5.0.dev0"


def implementation_revision() -> str:
    """Return an injected or source-tree Git revision without spawning Git."""
    injected = os.environ.get("OPCOMPASS_GIT_REVISION")
    if injected:
        return injected
    git_dir = Path(__file__).resolve().parent.parent / ".git"
    try:
        head = (git_dir / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            return (git_dir / head[5:]).read_text().strip()
        return head
    except OSError:
        return "unknown"
