from dataclasses import dataclass, field, replace
from typing import Any
import numpy as np


@dataclass(frozen=True)
class Memory:
    id: str
    seq: int
    state: str
    text: str | None
    vector: np.ndarray = field(repr=False, compare=False)
    probabilities: tuple[float, ...]
    hr_raw: float
    hr_normalized: float
    anchor: float
    selection_score: float
    confidence: float
    source: str
    created_at: float
    epsilon: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def data(self, include_vector=False):
        result = {k: v for k, v in self.__dict__.items() if k != "vector"}
        if include_vector:
            result["vector"] = self.vector.tolist()
        return result

    def with_state(self, state, **kwargs):
        return replace(self, state=state, **kwargs)


@dataclass(frozen=True)
class Hit:
    memory: Memory
    similarity: float
    score: float
    omega: float
    eureka: bool

    def data(self):
        return {"id": self.memory.id, "state": self.memory.state, "text": self.memory.text,
                "source": self.memory.source, "confidence": self.memory.confidence,
                "similarity": self.similarity, "score": self.score, "omega": self.omega,
                "eureka": self.eureka, "direction_only": self.memory.text is None}
