#!/usr/bin/env python3
"""Build and verify OpCompass distributions from a clean installation."""

import argparse
import glob
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def run(*command: str, cwd: Path = ROOT) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def installed_checks(venv: Path) -> None:
    bindir = "Scripts" if os.name == "nt" else "bin"
    python = venv / bindir / ("python.exe" if os.name == "nt" else "python")
    compass = venv / bindir / ("compass.exe" if os.name == "nt" else "compass")
    run(str(compass), "--version")
    run(str(compass), "list", "operators")
    run(str(compass), "list", "hardware")
    run(
        str(compass), "analyze", "matmul", "--hardware", "a100",
        "--dtype", "fp16", "--M", "128", "--N", "128", "--K", "128",
        "--format", "json",
    )
    run(str(python), "-c", (
        "from pathlib import Path; "
        "from opcompass.server import WEB_DIR, app; "
        "from opcompass.engine.solar_analyzer import HARDWARE_TO_SOLAR_ARCH; "
        "assert app.openapi()['info']['version']; "
        "assert (Path(WEB_DIR) / 'index.html').is_file(); "
        "assert all(Path(p).is_file() for p in HARDWARE_TO_SOLAR_ARCH.values())"
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        help="Keep artifacts in this directory instead of a temporary directory.",
    )
    args = parser.parse_args()

    temporary = None
    if args.output_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="opcompass-package-smoke-")
        output = Path(temporary.name)
    else:
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)

    dist = output / "dist"
    run(sys.executable, "-m", "build", "--outdir", str(dist))
    wheels = sorted(dist.glob("opcompass-*.whl"))
    sdists = sorted(dist.glob("opcompass-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError("expected exactly one wheel and one sdist")

    venv = output / "venv"
    run(sys.executable, "-m", "venv", str(venv))
    pip = venv / ("Scripts" if os.name == "nt" else "bin") / (
        "pip.exe" if os.name == "nt" else "pip"
    )
    run(str(pip), "install", str(wheels[0]))
    installed_checks(venv)

    source = output / "sdist-source"
    source.mkdir()
    with tarfile.open(sdists[0]) as archive:
        archive.extractall(source)
    roots = [Path(path) for path in glob.glob(str(source / "opcompass-*"))]
    if len(roots) != 1:
        raise RuntimeError("expected exactly one extracted sdist root")
    run(sys.executable, "-m", "build", "--wheel", "--outdir", str(output / "sdist-dist"), cwd=roots[0])
    print(f"Package smoke test passed; artifacts: {output}")

    if temporary is not None:
        temporary.cleanup()


if __name__ == "__main__":
    main()
