from dataclasses import asdict, dataclass, field
import math


DEFAULT_TOPICS = {
    "biology": "biologia célula proteína metabolismo membrana tecido organismo neurônio sinapse vida",
    "computing": "computação algoritmo programa memória vetor grafo aprendizado software rede dados",
    "physics": "física energia força movimento campo matéria onda entropia dinâmica calor",
    "methods": "método hipótese experimento evidência controle medida previsão teste resultado pesquisa",
}


@dataclass(frozen=True)
class Config:
    dimension: int = 384
    seed: int = 42
    topics: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_TOPICS))
    affinity_temperature: float = 8.0
    lambda_li: float = 0.5
    consolidation_min: float = 0.30
    consolidation_max: float = 0.82
    shadow_threshold: float = 0.30
    selector: str = "paper"  # paper, anchored, recency, random
    min_confidence: float = 0.0
    shadow_mode: str = "pointer"  # pointer or direction_only
    epsilon_max: float = 0.05
    shadow_half_life_steps: float = 1000.0
    urgency_alpha: float = 10.0
    entropy_prior: float = 0.65
    warmup_steps: float = 50.0
    resonance_threshold: float = 0.22
    retrieval_min_similarity: float = 0.05
    max_active: int = 256
    max_shadows: int = 1024
    max_pending: int = 512
    max_text_bytes: int = 32_768
    max_total_text_bytes: int = 4_194_304
    max_events: int = 2000
    max_proposals: int = 500
    max_degree: int = 8
    graph_min_similarity: float = 0.15
    cluster_similarity: float = 0.97
    rem_pairs: int = 16
    rem_attenuation: float = 0.25
    index: str = "exact"
    context_characters: int = 12_000

    def __post_init__(self):
        for name in ("dimension", "max_active", "max_shadows", "max_pending", "max_text_bytes",
                     "max_total_text_bytes", "max_events", "max_proposals", "context_characters"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.dimension < 8 or len(self.topics) < 2:
            raise ValueError("Use dimension >= 8 and at least two semantic prototypes")
        if any(not isinstance(k, str) or not isinstance(v, str) or not v.strip() for k, v in self.topics.items()):
            raise ValueError("Topic names and non-empty prototype texts must be strings")
        for name in ("affinity_temperature", "shadow_half_life_steps", "warmup_steps", "urgency_alpha"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("lambda_li", "consolidation_min", "consolidation_max", "shadow_threshold",
                     "min_confidence", "entropy_prior", "resonance_threshold", "retrieval_min_similarity",
                     "graph_min_similarity", "cluster_similarity", "rem_attenuation", "epsilon_max"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0 < self.epsilon_max < 1:
            raise ValueError("epsilon_max must be in (0, 1)")
        if not self.shadow_threshold <= self.consolidation_min <= self.consolidation_max:
            raise ValueError("Expected shadow_threshold <= consolidation_min <= consolidation_max")
        if self.selector not in {"paper", "anchored", "recency", "random"}:
            raise ValueError("Unknown selector")
        if self.shadow_mode not in {"pointer", "direction_only"}:
            raise ValueError("Unknown shadow mode")
        if self.index not in {"exact", "hnsw"}:
            raise ValueError("Unknown vector index")
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in (self.max_degree, self.rem_pairs)):
            raise ValueError("max_degree and rem_pairs must be non-negative")

    @property
    def consolidation_threshold(self):
        return self.consolidation_min + self.lambda_li * (self.consolidation_max - self.consolidation_min)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)
