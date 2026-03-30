from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass
class BenchmarkConfig:
    model: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    max_new_tokens: int = 64
    warmup_runs: int = 1
    measured_runs: int = 3
    output_dir: str = "results"

    def to_dict(self) -> dict:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")
