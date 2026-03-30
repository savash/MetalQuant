from __future__ import annotations

from metalquant.diagnose import diagnose_backend


def test_diagnose_prefers_tq4_for_healthy_kv_norms() -> None:
    diagnosis = diagnose_backend(model="mlx-community/Meta-Llama-3.1-8B-Instruct-8bit", kv_norm=18.0)

    assert diagnosis.recommended_backend == "tq4"
    assert diagnosis.confidence == "high"


def test_diagnose_prefers_fp16_outlier_for_large_kv_norms() -> None:
    diagnosis = diagnose_backend(model="mlx-community/Qwen2.5-7B-Instruct-4bit", kv_norm=274.0)

    assert diagnosis.recommended_backend == "fp16-outlier"
    assert diagnosis.confidence == "high"


def test_diagnose_uses_model_heuristics_when_norms_are_missing() -> None:
    diagnosis = diagnose_backend(model="mlx-community/Qwen2.5-7B-Instruct-4bit")

    assert diagnosis.recommended_backend == "fp16-outlier"
    assert diagnosis.confidence == "medium"


def test_diagnose_falls_back_to_int8_without_context() -> None:
    diagnosis = diagnose_backend()

    assert diagnosis.recommended_backend == "int8"
    assert diagnosis.confidence == "low"
