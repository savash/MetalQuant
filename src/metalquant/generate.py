from __future__ import annotations

from dataclasses import asdict, dataclass

from metalquant.diagnose import diagnose_backend


@dataclass
class GenerationPlan:
    requested_backend: str
    selected_backend: str
    confidence: str
    summary: str
    rationale: list[str]
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_backend_plan(
    requested_backend: str,
    model: str,
    kv_norm: float | None = None,
    calibration_path: str | None = None,
) -> GenerationPlan:
    if requested_backend != "auto":
        return GenerationPlan(
            requested_backend=requested_backend,
            selected_backend=requested_backend,
            confidence="explicit",
            summary="Backend selected explicitly by the user.",
            rationale=[],
            notes=[],
        )

    diagnosis = diagnose_backend(model=model, kv_norm=kv_norm)
    notes: list[str] = []
    selected_backend = diagnosis.recommended_backend

    if selected_backend == "fp16-outlier" and not calibration_path:
        selected_backend = "int8"
        notes.append(
            "Auto mode fell back to int8 because fp16-outlier requires calibration data."
        )
        notes.append(
            "Run `metalquant calibrate --model ...` and pass --calibration to enable fp16-outlier."
        )

    return GenerationPlan(
        requested_backend=requested_backend,
        selected_backend=selected_backend,
        confidence=diagnosis.confidence,
        summary=diagnosis.summary,
        rationale=diagnosis.rationale,
        notes=notes,
    )


def generate_text(
    model_name: str,
    prompt: str,
    backend: str = "auto",
    kv_norm: float | None = None,
    calibration_path: str | None = None,
    max_new_tokens: int = 128,
) -> tuple[GenerationPlan, str]:
    import mlx.core as mx
    from mlx_lm import load
    from metalquant.cache import make_cache
    from metalquant.calibrate import load_calibration

    calibration = load_calibration(calibration_path) if calibration_path else None
    plan = resolve_backend_plan(
        requested_backend=backend,
        model=model_name,
        kv_norm=kv_norm,
        calibration_path=calibration_path,
    )

    model, tokenizer = load(model_name)
    cache = make_cache(model, backend=plan.selected_backend, calibration=calibration)

    input_ids = mx.array(tokenizer.encode(prompt))[None]
    logits = model(input_ids, cache=cache)
    mx.eval(logits)

    token = mx.argmax(logits[:, -1, :], axis=-1)
    generated = [int(token.item())]

    for _ in range(max_new_tokens - 1):
        logits = model(token.reshape(1, 1), cache=cache)
        mx.eval(logits)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        token_id = int(token.item())
        generated.append(token_id)
        if token_id == tokenizer.eos_token_id:
            break

    return plan, tokenizer.decode(generated)
