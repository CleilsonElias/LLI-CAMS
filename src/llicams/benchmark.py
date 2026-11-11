"""Delayed-cue retrieval of random factual codes. No LLM and no creativity assessment."""
from dataclasses import replace
import json
import numpy as np
from .config import Config
from .engine import Engine
from .math import unit


class PrototypeEncoder:
    def __init__(self, dimension=24):
        self.dimension = dimension
        self.fingerprint = f"benchmark-prototypes-v1:{dimension}"

    def encode(self, text):
        vector = np.zeros(self.dimension)
        vector[int(text.removeprefix("topic-"))] = 1
        return vector


def run_benchmark(seeds=5, observations=160, queries=48):
    if seeds < 1 or observations < 16 or not 1 <= queries <= observations:
        raise ValueError("Use seeds >= 1, observations >= 16 and 1 <= queries <= observations")
    conditions = ["paper", "paper_no_urgency", "anchored", "recency", "random", "direction_only"]
    runs = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed + 100)
        vectors, texts, values = [], [], []
        for i in range(observations):
            v = rng.normal(0, 0.12, 24)
            domains = rng.choice(4, size=int(rng.choice([1, 2, 3])), replace=False)
            v[domains] += 0.7
            vectors.append(unit(v))
            value = f"V-{int(rng.integers(100000,999999))}"
            values.append(value)
            texts.append(f"Registro sintético K-{i:04d}: código factual {value}.")
        chosen = rng.choice(observations, size=queries, replace=False)
        query_vectors = {int(i): unit(vectors[i] + rng.normal(0, 0.025, 24)) for i in chosen}
        for condition in conditions:
            config = Config(dimension=24, seed=seed, topics={str(i): f"topic-{i}" for i in range(4)},
                            max_active=24, max_shadows=48, max_pending=64, shadow_threshold=0.12,
                            consolidation_min=0.30, consolidation_max=0.82,
                            selector=condition if condition in {"anchored", "recency", "random"} else "paper",
                            shadow_mode="direction_only" if condition == "direction_only" else "pointer")
            with Engine(config=config, encoder=PrototypeEncoder()) as engine:
                for i, (v, text) in enumerate(zip(vectors, texts)):
                    engine.observe(text, vector=v, memory_id=f"m-{i:04d}", source="synthetic:benchmark")
                    if (i + 1) % 32 == 0:
                        engine.sleep()
                engine.sleep()
                correct, eureka = 0, 0
                use_urgency = condition not in {"paper_no_urgency", "recency", "random"}
                for i in chosen:
                    result = engine.retrieve(vector=query_vectors[int(i)], k=1, include_pending=False,
                                             use_urgency=use_urgency)
                    hit = result["hits"][0] if result["hits"] else None
                    correct += bool(hit and hit.memory.text and values[i] in hit.memory.text)
                    eureka += bool(hit and hit.eureka)
                stats = engine.stats()
                runs.append({"condition": condition, "seed": seed, "correct": correct, "queries": queries,
                             "accuracy": correct / queries, "shadow_hits": eureka,
                             "active": stats["active"], "shadows": stats["shadow"],
                             "text_bytes": stats["text_bytes"], "vector_bytes": stats["vector_bytes"]})
    summary = []
    for condition in conditions:
        records = [r for r in runs if r["condition"] == condition]
        accuracy = [r["accuracy"] for r in records]
        summary.append({"condition": condition, "mean_accuracy": float(np.mean(accuracy)),
                        "std_across_seeds": float(np.std(accuracy, ddof=1)) if seeds > 1 else 0.0,
                        "runs": len(records)})
    return {"benchmark": "synthetic-delayed-code-retrieval-v1", "seeds": seeds,
            "observations_per_seed": observations, "queries_per_seed": queries,
            "capacity": {"active_items": 24, "shadow_items": 48},
            "limitations": ["Supplied synthetic vectors, not a natural-language semantic benchmark",
                            "Equal item caps; total bytes are reported and are not equalized",
                            "No transformer training, consciousness measure or creativity assessment",
                            "Queries sampled independently of H_R; identifiers are used only for evaluation"],
            "summary": summary, "runs": runs}
