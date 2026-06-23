"""CLI entry point for OpCompass — ``compass`` command.

Usage::

    compass list operators
    compass list hardware
    compass info matmul
    compass analyze matmul --hardware a100 --dtype fp16 --M 4096 --N 4096 --K 4096
    compass sweep matmul --hardware a100,h100 --M 1024,2048,4096 --K 1024,2048,4096
"""

from __future__ import annotations

import sys
from typing import Any

import click

from opcompass.registry import discover_hardware, discover_operators, get_hardware, get_operator
from opcompass.models import AnalysisMode, DataType, PipelineConfig
from opcompass.engine.analyzer import Analyzer
from opcompass.engine.result import format_result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_dtype(dtype_str: str) -> DataType:
    """Resolve a dtype string to a DataType enum member."""
    try:
        return DataType(dtype_str.lower())
    except ValueError:
        valid = ", ".join(dt.value for dt in DataType)
        raise click.BadParameter(f"Unknown dtype '{dtype_str}'. Valid: {valid}")


def _resolve_mode(mode_str: str) -> AnalysisMode:
    """Resolve mode string."""
    try:
        return AnalysisMode(mode_str.lower())
    except ValueError:
        valid = ", ".join(m.value for m in AnalysisMode)
        raise click.BadParameter(f"Unknown mode '{mode_str}'. Valid: {valid}")


def _parse_dim_args(ctx: click.Context) -> dict[str, int]:
    """Extract dimension parameters from leftover ``--<dim> <value>`` args.

    Any option that isn't a recognised Click option is treated as a dimension.
    """
    dims: dict[str, int] = {}
    args = ctx.args
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            key = arg[2:]
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                try:
                    dims[key] = int(args[i + 1])
                except ValueError:
                    dims[key] = args[i + 1]  # allow non-int? no — dims must be int
                i += 2
            else:
                i += 1  # flag without value, skip
        else:
            i += 1
    return dims


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(version="0.1.0", prog_name="compass")
def main():
    """OpCompass — SOL theoretical peak performance estimator for GPU operators."""


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@main.group("list")
def list_group():
    """List available operators or hardware targets."""


@list_group.command("operators")
def list_operators():
    """List all registered operators."""
    ops = discover_operators()
    if not ops:
        click.echo("No operators found.")
        return
    click.echo(f"{'Name':<20} Description")
    click.echo("-" * 60)
    for name, cls in sorted(ops.items()):
        click.echo(f"{name:<20} {cls.description}")


@list_group.command("hardware")
def list_hardware():
    """List all registered hardware targets."""
    hw = discover_hardware()
    if not hw:
        click.echo("No hardware targets found.")
        return
    click.echo(f"{'Name':<10} {'Vendor':<10} Description")
    click.echo("-" * 60)
    for name, cls in sorted(hw.items()):
        click.echo(f"{name:<10} {cls.vendor:<10} {cls.description}")


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

@main.command("info")
@click.argument("operator_name")
def info_cmd(operator_name: str):
    """Show detailed information about an operator."""
    try:
        cls = get_operator(operator_name)
    except KeyError as e:
        click.echo(str(e), err=True)
        sys.exit(1)

    inst = cls()
    click.echo(f"Operator: {inst.name}")
    click.echo(f"Description: {inst.description}")
    click.echo()
    click.echo("Dimension parameters:")
    for dim, desc in inst.param_dims.items():
        click.echo(f"  {dim:<10} {desc}")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

@main.command("analyze", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.option("--hardware", "-H", required=True, help="Hardware target (e.g. a100)")
@click.option("--dtype", "-d", default="fp16", help="Data type (default: fp16)")
@click.option("--mode", "-m", default="hierarchy", help="Analysis mode: simple, hierarchy, pipeline")
@click.option("--format", "-f", "fmt", default="table", help="Output format: table, json, csv")
@click.option("--async-copy/--no-async-copy", default=True, help="Enable/disable async copy (pipeline mode)")
@click.option("--sparsity/--no-sparsity", default=False, help="Enable/disable 2:4 sparsity (pipeline mode)")
@click.argument("operator_name")
@click.pass_context
def analyze_cmd(
    ctx: click.Context,
    hardware: str,
    dtype: str,
    mode: str,
    fmt: str,
    async_copy: bool,
    sparsity: bool,
    operator_name: str,
):
    """Analyze SOL performance for an operator.

    Pass dimension values as extra flags, e.g.:

        compass analyze matmul --hardware a100 --M 4096 --N 4096 --K 4096
    """
    # Resolve operator & hardware
    try:
        op_cls = get_operator(operator_name)
    except KeyError as e:
        click.echo(str(e), err=True)
        sys.exit(1)

    try:
        hw_cls = get_hardware(hardware)
    except KeyError as e:
        click.echo(str(e), err=True)
        sys.exit(1)

    # Parse dimensions from extra args
    dims = _parse_dim_args(ctx)

    if not dims:
        click.echo(
            "Error: No dimension arguments provided. "
            "Use e.g. --M 4096 --N 4096 --K 4096",
            err=True,
        )
        sys.exit(1)

    resolved_dtype = _resolve_dtype(dtype)
    resolved_mode = _resolve_mode(mode)

    # Build pipeline config if in pipeline mode
    pipeline_config = None
    if resolved_mode == AnalysisMode.PIPELINE:
        pipeline_config = PipelineConfig(
            async_copy_enabled=async_copy,
            sparsity_2_4_enabled=sparsity,
        )

    op_inst = op_cls()
    hw_inst = hw_cls()

    analyzer = Analyzer()
    result = analyzer.analyze(
        op_inst, hw_inst, resolved_dtype, mode=resolved_mode,
        pipeline_config=pipeline_config, **dims
    )

    click.echo(format_result(result, fmt=fmt))


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

@main.command("sweep", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.option("--hardware", "-H", required=True, help="Hardware target(s), comma-separated (e.g. a100,h100)")
@click.option("--dtype", "-d", default="fp16", help="Data type")
@click.option("--mode", "-m", default="hierarchy", help="Analysis mode")
@click.option("--format", "-f", "fmt", default="table", help="Output format: table, json, csv")
@click.argument("operator_name")
@click.pass_context
def sweep_cmd(
    ctx: click.Context,
    hardware: str,
    dtype: str,
    mode: str,
    fmt: str,
    operator_name: str,
):
    """Sweep over multiple dimensions / hardware targets.

    Dimensions that receive comma-separated values are swept::

        compass sweep matmul --hardware a100,h100 --M 1024,2048,4096 --K 1024,2048,4096
    """
    try:
        op_cls = get_operator(operator_name)
    except KeyError as e:
        click.echo(str(e), err=True)
        sys.exit(1)

    hw_names = [h.strip() for h in hardware.split(",")]
    hw_instances = []
    for name in hw_names:
        try:
            hw_instances.append(get_hardware(name)())
        except KeyError as e:
            click.echo(str(e), err=True)
            sys.exit(1)

    resolved_dtype = _resolve_dtype(dtype)
    resolved_mode = _resolve_mode(mode)

    # Parse dims and identify sweep axes
    raw_dims = _parse_dim_args(ctx)
    sweep_axes: dict[str, list[int]] = {}
    fixed_dims: dict[str, int] = {}

    for k, v in raw_dims.items():
        if "," in str(v):
            sweep_axes[k] = [int(x.strip()) for x in str(v).split(",")]
        else:
            fixed_dims[k] = int(v)

    if not sweep_axes:
        # Nothing to sweep — just run a single analysis
        op_inst = op_cls()
        analyzer = Analyzer()
        for hw_inst in hw_instances:
            result = analyzer.analyze(
                op_inst, hw_inst, resolved_dtype, mode=resolved_mode, **fixed_dims
            )
            click.echo(format_result(result, fmt=fmt))
            click.echo()
        return

    # Cartesian product sweep
    op_inst = op_cls()
    analyzer = Analyzer()
    results = []

    from itertools import product

    axis_names = list(sweep_axes.keys())
    axis_values = [sweep_axes[k] for k in axis_names]

    for combo in product(*axis_values):
        combo_dims = dict(zip(axis_names, combo))
        all_dims = {**fixed_dims, **combo_dims}
        for hw_inst in hw_instances:
            result = analyzer.analyze(
                op_inst, hw_inst, resolved_dtype, mode=resolved_mode, **all_dims
            )
            results.append(result)

    # Output
    if fmt == "json":
        import json
        click.echo(json.dumps(
            [_result_to_dict(r) for r in results], indent=2, ensure_ascii=False
        ))
    elif fmt == "csv":
        click.echo("operator,hardware," + ",".join(axis_names) + "," + _csv_header())
        for r in results:
            vals = ",".join(str(r.shapes.get(k, "")) for k in axis_names)
            click.echo(f"{r.operator},{r.hardware},{vals},{_csv_row(r)}")
    else:
        # Table
        header = f"{'Hardware':<10} " + " ".join(f"{k:<12}" for k in axis_names) + f" {'SOL(µs)':>10} {'TFLOPS':>8} {'Bottleneck':>14}"
        click.echo(header)
        click.echo("-" * len(header))
        for r in results:
            dim_str = " ".join(f"{r.shapes.get(k, ''):<12}" for k in axis_names)
            click.echo(
                f"{r.hardware:<10} {dim_str} {r.sol_time_s*1e6:>10.1f} {r.sol_tflops:>8.1f} {r.bottleneck:>14}"
            )


# ---------------------------------------------------------------------------
# Helpers for sweep output
# ---------------------------------------------------------------------------

def _result_to_dict(result) -> dict:
    from opcompass.engine.result import _result_to_dict
    return _result_to_dict(result)


def _csv_header() -> str:
    return "sol_time_us,sol_tflops,bottleneck"


def _csv_row(result) -> str:
    return f"{result.sol_time_s*1e6:.2f},{result.sol_tflops:.2f},{result.bottleneck}"


if __name__ == "__main__":
    main()
