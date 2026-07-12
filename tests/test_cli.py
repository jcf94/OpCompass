"""Test CLI-visible validation behavior."""

from click.testing import CliRunner

from opcompass.cli import main


def test_cli_analyze_rejects_invalid_pipeline_block_granularity():
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "analyze",
            "--hardware", "a100",
            "--dtype", "fp16",
            "--mode", "pipeline",
            "--block-m", "63",
            "--block-n", "64",
            "--block-k", "16",
            "matmul",
            "--M", "4096",
            "--N", "4096",
            "--K", "4096",
        ],
    )

    assert result.exit_code == 1
    assert "multiple of 16" in result.output


def test_cli_sweep_rejects_invalid_pipeline_block_granularity():
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "sweep",
            "--hardware", "a100",
            "--dtype", "fp16",
            "--mode", "pipeline",
            "--block-m", "64",
            "--block-n", "65",
            "--block-k", "16",
            "matmul",
            "--M", "4096",
            "--N", "4096",
            "--K", "4096",
        ],
    )

    assert result.exit_code == 1
    assert "multiple of 8" in result.output


def test_cli_analyze_accepts_pipeline_stage_and_warp_overrides():
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "analyze",
            "--hardware", "a100",
            "--dtype", "fp16",
            "--mode", "pipeline",
            "--format", "json",
            "--block-m", "64",
            "--block-n", "128",
            "--block-k", "16",
            "--stage-count", "3",
            "--warp-count", "8",
            "matmul",
            "--M", "1024",
            "--N", "1024",
            "--K", "1024",
        ],
    )

    assert result.exit_code == 0
    assert '"stage_count": 3' in result.output
    assert '"warp_count": 8' in result.output
    assert '"pipeline_candidates"' in result.output


def test_cli_analyze_rejects_missing_required_dimension():
    result = CliRunner().invoke(
        main,
        ["analyze", "--hardware", "a100", "matmul", "--M", "128", "--N", "128"],
    )

    assert result.exit_code == 1
    assert "missing required parameter 'K'" in result.output


def test_cli_analyze_rejects_unknown_dimension():
    result = CliRunner().invoke(
        main,
        [
            "analyze", "--hardware", "a100", "matmul",
            "--M", "128", "--N", "128", "--K", "128", "--batch", "2",
        ],
    )

    assert result.exit_code == 1
    assert "unknown parameter 'batch'" in result.output


def test_cli_pipeline_fallback_is_explicit_in_json():
    result = CliRunner().invoke(
        main,
        [
            "analyze", "--hardware", "a100", "--mode", "pipeline",
            "--format", "json", "reduction", "--N", "4096", "--D", "256",
        ],
    )

    assert result.exit_code == 0
    assert '"requested_mode": "pipeline"' in result.output
    assert '"executed_mode": "hierarchy_roofline"' in result.output
    assert '"reason_code": "pipeline_model_unavailable"' in result.output


def test_cli_strict_pipeline_rejects_fallback():
    result = CliRunner().invoke(
        main,
        [
            "analyze", "--hardware", "a100", "--mode", "pipeline", "--strict",
            "reduction", "--N", "4096", "--D", "256",
        ],
    )

    assert result.exit_code == 1
    assert "has no pipeline model" in result.output


def test_cli_trace_is_opt_in_and_bounded():
    args = [
        "analyze", "--hardware", "a100", "--mode", "pipeline",
        "--format", "json", "matmul", "--M", "256", "--N", "256", "--K", "256",
    ]
    compact = CliRunner().invoke(main, args)
    traced = CliRunner().invoke(main, args[:7] + ["--trace", "--trace-limit", "2"] + args[7:])

    assert compact.exit_code == 0
    assert '"sub_ops"' not in compact.output
    assert traced.exit_code == 0
    assert traced.output.count('"pipeline_stage"') == 2


def test_cli_rejects_unknown_output_format():
    result = CliRunner().invoke(
        main,
        [
            "analyze", "--hardware", "a100", "--format", "yaml",
            "matmul", "--M", "128", "--N", "128", "--K", "128",
        ],
    )

    assert result.exit_code == 2
    assert "Invalid value for '--format'" in result.output


def test_cli_reports_unsupported_dtype_without_non_finite_output():
    result = CliRunner().invoke(
        main,
        [
            "analyze", "--hardware", "a100", "--dtype", "fp8",
            "matmul", "--M", "128", "--N", "128", "--K", "128",
        ],
    )

    assert result.exit_code == 1
    assert "does not support dtype 'fp8'" in result.output
    assert "inf" not in result.output.lower()
