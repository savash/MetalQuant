from __future__ import annotations

import json

from metalquant.cli import main


def test_cli_diagnose_prints_human_readable_output(capsys) -> None:
    exit_code = main(["diagnose", "--model", "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Recommended backend: tq4" in captured.out
    assert "Confidence: medium" in captured.out


def test_cli_diagnose_prints_json(capsys) -> None:
    exit_code = main(["diagnose", "--kv-norm", "274", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["recommended_backend"] == "fp16-outlier"
    assert payload["confidence"] == "high"
