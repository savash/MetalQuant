from __future__ import annotations

import json
from pathlib import Path

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


def test_cli_calibrate_wraps_benchmark_script(monkeypatch) -> None:
    captured = {}

    def fake_run(command, cwd, env, check):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["env"] = env
        captured["check"] = check
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr("metalquant.cli.subprocess.run", fake_run)

    exit_code = main(
        [
            "calibrate",
            "--model",
            "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "--n-outlier",
            "16",
            "--out",
            "results/custom-calibration.json",
        ]
    )

    assert exit_code == 0
    assert captured["command"][1].endswith("benchmarks/run_calibrate.py")
    assert captured["command"][2:] == [
        "--model",
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "--n-outlier",
        "16",
        "--out",
        "results/custom-calibration.json",
    ]
    assert Path(captured["cwd"]).name == "MetalQuant"
    assert captured["check"] is False
    assert "PYTHONPATH" in captured["env"]


def test_cli_benchmark_wraps_experiment_script(monkeypatch) -> None:
    captured = {}

    def fake_run(command, cwd, env, check):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["env"] = env
        captured["check"] = check
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr("metalquant.cli.subprocess.run", fake_run)

    exit_code = main(
        [
            "benchmark",
            "--model",
            "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
            "--cache-backend",
            "tq4",
            "--max-new-tokens",
            "32",
            "--calibration",
            "results/calibration.json",
            "--out",
            "results/tq4.json",
        ]
    )

    assert exit_code == 0
    assert captured["command"][1].endswith("benchmarks/run_experiment.py")
    assert captured["command"][2:] == [
        "--model",
        "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
        "--cache-backend",
        "tq4",
        "--max-new-tokens",
        "32",
        "--out",
        "results/tq4.json",
        "--calibration",
        "results/calibration.json",
    ]
    assert Path(captured["cwd"]).name == "MetalQuant"
    assert captured["check"] is False
    assert "PYTHONPATH" in captured["env"]
