from __future__ import annotations

import json

from metalquant.config import BenchmarkConfig


def test_benchmark_config_to_dict_uses_defaults() -> None:
    config = BenchmarkConfig()

    assert config.to_dict() == {
        "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "max_new_tokens": 64,
        "warmup_runs": 1,
        "measured_runs": 3,
        "output_dir": "results",
    }


def test_benchmark_config_write_json_writes_trailing_newline(tmp_path) -> None:
    path = tmp_path / "benchmark-config.json"
    config = BenchmarkConfig(model="demo/model", max_new_tokens=12, measured_runs=5)

    config.write_json(path)

    assert json.loads(path.read_text()) == {
        "model": "demo/model",
        "max_new_tokens": 12,
        "warmup_runs": 1,
        "measured_runs": 5,
        "output_dir": "results",
    }
    assert path.read_text().endswith("\n")
