from typing import Optional

from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    model: str
    model_params: Optional[dict]