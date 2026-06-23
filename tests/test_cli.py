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
