from __future__ import annotations

from metalquant.generate import resolve_backend_plan


def test_resolve_backend_plan_keeps_explicit_backend() -> None:
    plan = resolve_backend_plan(
        requested_backend="tq4",
        model="mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
    )

    assert plan.selected_backend == "tq4"
    assert plan.confidence == "explicit"


def test_resolve_backend_plan_uses_fp16_outlier_when_calibration_exists() -> None:
    plan = resolve_backend_plan(
        requested_backend="auto",
        model="mlx-community/Qwen2.5-7B-Instruct-4bit",
        kv_norm=274.0,
        calibration_path="results/calibration.json",
    )

    assert plan.selected_backend == "fp16-outlier"
    assert plan.confidence == "high"
    assert plan.notes == []


def test_resolve_backend_plan_falls_back_to_int8_without_calibration() -> None:
    plan = resolve_backend_plan(
        requested_backend="auto",
        model="mlx-community/Qwen2.5-7B-Instruct-4bit",
        kv_norm=274.0,
        calibration_path=None,
    )

    assert plan.selected_backend == "int8"
    assert plan.confidence == "high"
    assert plan.notes
