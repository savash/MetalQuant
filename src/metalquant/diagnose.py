from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Diagnosis:
    model: str | None
    kv_norm: float | None
    recommended_backend: str
    confidence: str
    summary: str
    rationale: list[str]
    next_steps: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def diagnose_backend(model: str | None = None, kv_norm: float | None = None) -> Diagnosis:
    normalized_model = model.lower() if model else None
    rationale: list[str] = []
    next_steps: list[str] = []

    if kv_norm is not None:
        rationale.append(f"Observed KV norm: {kv_norm:.1f}")
        if kv_norm < 30:
            rationale.append("KV norms are in the healthy range for standard TurboQuant.")
            next_steps.extend(
                [
                    "Start with tq4 for the best quality/compression balance.",
                    "Try tq2 only after confirming generation quality on your prompts.",
                ]
            )
            return Diagnosis(
                model=model,
                kv_norm=kv_norm,
                recommended_backend="tq4",
                confidence="high",
                summary="Healthy KV norms suggest standard TurboQuant should be safe.",
                rationale=rationale,
                next_steps=next_steps,
            )

        if kv_norm < 50:
            rationale.append("KV norms are borderline: TurboQuant may work, but quality risk is higher.")
            next_steps.extend(
                [
                    "Compare tq4 against int8 on a small prompt set before using it broadly.",
                    "If quality drifts, fall back to int8 or run calibration for fp16-outlier.",
                ]
            )
            return Diagnosis(
                model=model,
                kv_norm=kv_norm,
                recommended_backend="int8",
                confidence="medium",
                summary="Borderline KV norms favor a conservative starting point.",
                rationale=rationale,
                next_steps=next_steps,
            )

        rationale.append("KV norms are too large for standard TurboQuant to be a safe default.")
        next_steps.extend(
            [
                "Run calibration first, then use fp16-outlier.",
                "If you need the safest immediate fallback, use int8.",
            ]
        )
        return Diagnosis(
            model=model,
            kv_norm=kv_norm,
            recommended_backend="fp16-outlier",
            confidence="high",
            summary="Large KV norms indicate a model that likely needs the outlier-aware path.",
            rationale=rationale,
            next_steps=next_steps,
        )

    if normalized_model:
        if "4bit" in normalized_model:
            rationale.append("Model name suggests 4-bit weight quantization.")
            next_steps.extend(
                [
                    "Run `metalquant diagnose --model ... --kv-norm <measured-norm>` for a stronger recommendation.",
                    "If norms are inflated, calibrate and use fp16-outlier.",
                ]
            )
            return Diagnosis(
                model=model,
                kv_norm=None,
                recommended_backend="fp16-outlier",
                confidence="medium",
                summary="4-bit models often need the outlier-aware path.",
                rationale=rationale,
                next_steps=next_steps,
            )

        if "8bit" in normalized_model or "int8" in normalized_model:
            rationale.append("Model name suggests 8-bit weights, which are usually healthier for TurboQuant.")
            next_steps.extend(
                [
                    "Use tq4 as the default starting point.",
                    "Measure KV norms if you want higher confidence before trying tq2.",
                ]
            )
            return Diagnosis(
                model=model,
                kv_norm=None,
                recommended_backend="tq4",
                confidence="medium",
                summary="8-bit models are often the best fit for standard TurboQuant.",
                rationale=rationale,
                next_steps=next_steps,
            )

        rationale.append("Model name alone is not enough to trust aggressive compression.")
        next_steps.extend(
            [
                "Measure KV norms before enabling TurboQuant.",
                "Start with int8 if you need a safe default today.",
            ]
        )
        return Diagnosis(
            model=model,
            kv_norm=None,
            recommended_backend="int8",
            confidence="low",
            summary="Without KV norms, a conservative backend recommendation is safer.",
            rationale=rationale,
            next_steps=next_steps,
        )

    rationale.append("No model metadata or KV norms were provided.")
    next_steps.extend(
        [
            "Pass --model for a heuristic recommendation.",
            "Pass --kv-norm for a stronger recommendation based on measured behavior.",
        ]
    )
    return Diagnosis(
        model=None,
        kv_norm=None,
        recommended_backend="int8",
        confidence="low",
        summary="Not enough information for model-specific advice, so the safest default is int8.",
        rationale=rationale,
        next_steps=next_steps,
    )
