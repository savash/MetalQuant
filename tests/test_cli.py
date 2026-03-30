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


def test_cli_calibrate_uses_model_specific_default_output(monkeypatch) -> None:
    captured = {}

    def fake_run(command, cwd, env, check):
        captured["command"] = command
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr("metalquant.cli.subprocess.run", fake_run)

    exit_code = main(
        [
            "calibrate",
            "--model",
            "mlx-community/Qwen2.5-7B-Instruct-4bit",
        ]
    )

    assert exit_code == 0
    assert captured["command"][2:] == [
        "--model",
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "--n-outlier",
        "32",
        "--out",
        "results/calibration-qwen2-5-7b-instruct-4bit.json",
    ]


def test_cli_generate_uses_auto_plan(monkeypatch, capsys) -> None:
    def fake_generate_text(model_name, prompt, backend, kv_norm, calibration_path, max_new_tokens):
        return (
            type(
                "Plan",
                (),
                {
                    "selected_backend": "tq4",
                    "confidence": "medium",
                    "summary": "8-bit models are often the best fit for standard TurboQuant.",
                    "rationale": ["Model name suggests 8-bit weights."],
                    "notes": [],
                    "to_dict": lambda self: {
                        "requested_backend": backend,
                        "selected_backend": "tq4",
                        "confidence": "medium",
                        "summary": "8-bit models are often the best fit for standard TurboQuant.",
                        "rationale": ["Model name suggests 8-bit weights."],
                        "notes": [],
                    },
                },
            )(),
            "Generated text",
        )

    monkeypatch.setattr("metalquant.cli.generate_text", fake_generate_text)

    exit_code = main(
        [
            "generate",
            "--model",
            "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
            "--prompt",
            "Hello",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Selected backend: tq4" in captured.out
    assert "Output:" in captured.out
    assert "Generated text" in captured.out


def test_cli_generate_prints_json(monkeypatch, capsys) -> None:
    def fake_generate_text(model_name, prompt, backend, kv_norm, calibration_path, max_new_tokens):
        return (
            type(
                "Plan",
                (),
                {
                    "selected_backend": "int8",
                    "confidence": "low",
                    "summary": "Fallback plan.",
                    "rationale": [],
                    "notes": ["Auto mode fell back to int8."],
                    "to_dict": lambda self: {
                        "requested_backend": backend,
                        "selected_backend": "int8",
                        "confidence": "low",
                        "summary": "Fallback plan.",
                        "rationale": [],
                        "notes": ["Auto mode fell back to int8."],
                    },
                },
            )(),
            "Generated text",
        )

    monkeypatch.setattr("metalquant.cli.generate_text", fake_generate_text)

    exit_code = main(
        [
            "generate",
            "--model",
            "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "--prompt",
            "Hello",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["plan"]["selected_backend"] == "int8"
    assert payload["output"] == "Generated text"


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


def test_cli_benchmark_auto_selects_backend_and_default_output(monkeypatch, capsys) -> None:
    captured = {}

    def fake_run(command, cwd, env, check):
        captured["command"] = command
        return type("Completed", (), {"returncode": 0})()

    def fake_resolve_backend_plan(requested_backend, model, kv_norm, calibration_path):
        return type(
            "Plan",
            (),
            {
                "selected_backend": "tq4",
                "confidence": "high",
                "summary": "Healthy KV norms suggest standard TurboQuant should be safe.",
                "notes": [],
            },
        )()

    monkeypatch.setattr("metalquant.cli.subprocess.run", fake_run)
    monkeypatch.setattr("metalquant.cli.resolve_backend_plan", fake_resolve_backend_plan)

    exit_code = main(
        [
            "benchmark",
            "--model",
            "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
            "--kv-norm",
            "18",
        ]
    )

    captured_output = capsys.readouterr()

    assert exit_code == 0
    assert "Selected backend: tq4" in captured_output.out
    assert captured["command"][2:] == [
        "--model",
        "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
        "--cache-backend",
        "tq4",
        "--max-new-tokens",
        "64",
        "--out",
        "results/meta-llama-3-1-8b-instruct-8bit-tq4.json",
    ]
