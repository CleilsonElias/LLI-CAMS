from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import threading
import time
import uuid
import numpy as np
from .config import Config
from .encoders import HashEncoder
from .index import VectorIndex
from .math import unit, cluster_affinities, relational_entropy, normalized_relational_entropy, historical_baseline, urgency
from .storage import Store, ConflictError, json_text
from .types import Memory, Hit
from .sleep import plan, bound_states, build_graph


class Engine:
    def __init__(self, path=":memory:", config=None, encoder=None, replay=None, rem=None):
        # Deep-copy topic mapping; frozen dataclasses alone do not freeze dictionaries.
        self.config = Config.from_dict((config or Config()).to_dict())
        self.encoder = encoder or HashEncoder(self.config.dimension, self.config.seed)
        if self.encoder.dimension != self.config.dimension:
            raise ValueError("Encoder and configuration dimensions differ")
        self.prototypes = np.stack([unit(self.encoder.encode(text), self.config.dimension)
                                    for text in self.config.topics.values()])
        self.store = Store(path, self.config, self.encoder.fingerprint)
        self.replay, self.rem = replay, rem
        self.sleep_lock = threading.Lock()
        self.cache_lock = threading.Lock()
        self._cached_version = None
        self._cached_indices = None

    def features(self, vector):
        vector = unit(vector, self.config.dimension)
        p, anchor = cluster_affinities(vector, self.prototypes, self.config.affinity_temperature)
        raw, normalized = relational_entropy(p), normalized_relational_entropy(p)
        score = normalized if self.config.selector != "anchored" else normalized * anchor
        return vector, p, anchor, raw, normalized, score

    def observe(self, text, *, source="manual", confidence=1.0, metadata=None, memory_id=None, vector=None):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be non-empty")
        if len(text.encode()) > self.config.max_text_bytes:
            raise ValueError("Observation exceeds max_text_bytes")
        if not isinstance(source, str) or len(source.encode()) > 4096:
            raise ValueError("Source must be a string of at most 4096 bytes")
        if not isinstance(confidence, (float, int)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Confidence must be in [0,1]")
        metadata = json.loads(json_text(metadata or {}))
        if not isinstance(metadata, dict) or len(json_text(metadata).encode()) > 8192:
            raise ValueError("Metadata must be a JSON object of at most 8192 bytes")
        if "pinned" in metadata and not isinstance(metadata["pinned"], bool):
            raise ValueError("metadata.pinned must be boolean")
        memory_id = memory_id or str(uuid.uuid4())
        if not isinstance(memory_id, str) or not memory_id or len(memory_id) > 200:
            raise ValueError("Invalid memory id")
        v, p, anchor, raw, normalized, score = self.features(vector if vector is not None else self.encoder.encode(text))
        record = Memory(memory_id, 0, "pending", text, v, tuple(map(float, p)), raw, normalized,
                        anchor, score, float(confidence), source, time.time(), metadata=metadata)
        return self.store.append(record)

    def _indices(self, view):
        with self.cache_lock:
            if self._cached_version != view["version"]:
                self._cached_indices = (
                    VectorIndex([m for m in view["records"] if m.state == "active"], self.config.index, self.config.seed),
                    VectorIndex([m for m in view["records"] if m.state == "shadow"], self.config.index, self.config.seed))
                self._cached_version = view["version"]
            return self._cached_indices

    def retrieve(self, query=None, *, vector=None, k=5, include_pending=True, use_urgency=True):
        if not isinstance(k, int) or not 1 <= k <= 100:
            raise ValueError("k must be between 1 and 100")
        v, p, anchor, raw, normalized, score = self.features(vector if vector is not None else self.encoder.encode(query))
        view = self.store.view()
        baseline = historical_baseline(view["hr_total"], view["hr_count"],
                                       self.config.entropy_prior, self.config.warmup_steps)
        omega = urgency(normalized, baseline, self.config.urgency_alpha) if use_urgency else 1.0
        active_index, shadow_index = self._indices(view)
        hits = []
        for record, similarity in active_index.search(v, k):
            if similarity >= self.config.retrieval_min_similarity:
                hits.append(Hit(record, similarity, similarity, omega, False))
        # epsilon does not participate in resonance: faithfully follows the addendum.
        for record, similarity in shadow_index.search(v, max(k, 20)):
            resonance = similarity * omega
            if similarity >= self.config.retrieval_min_similarity and resonance > self.config.resonance_threshold:
                hits.append(Hit(record, similarity, resonance, omega, True))
        if include_pending:
            pending_index = VectorIndex([m for m in view["records"] if m.state == "pending"])
            for record, similarity in pending_index.search(v, k):
                if similarity >= self.config.retrieval_min_similarity:
                    hits.append(Hit(record, similarity, similarity, omega, False))
        hits.sort(key=lambda h: (-h.score, h.memory.id))
        return {"version": view["version"], "weight_artifact": view["weight_artifact"],
                "hr_raw": raw, "hr_normalized": normalized, "baseline": baseline, "omega": omega,
                "hits": hits[:k]}

    def sleep(self):
        if not self.sleep_lock.acquire(blocking=False):
            raise RuntimeError("A sleep cycle is already being prepared by this Engine")
        try:
            view = self.store.view()
            records, edges, proposals, artifact, report = plan(view, self.config, self.replay, self.rem)
            # Validate optional indices before publication; a missing backend cannot break live state.
            VectorIndex([m for m in records if m.state == "active"], self.config.index, self.config.seed)
            VectorIndex([m for m in records if m.state == "shadow"], self.config.index, self.config.seed)
            version = self.store.commit(view["version"], view["seq"], records, edges, report, proposals, artifact)
            return {**report, "version": version}
        finally:
            self.sleep_lock.release()

    def promote(self, memory_id, evidence_note):
        if not isinstance(evidence_note, str) or not evidence_note.strip() or len(evidence_note) > 4000:
            raise ValueError("Promotion requires an explicit evidence note (at most 4000 characters)")
        view = self.store.view()
        selected = next((m for m in view["records"] if m.id == memory_id and m.state == "shadow"), None)
        if selected is None:
            raise ValueError("Shadow id not found")
        if selected.text is None:
            raise ValueError("A direction-only shadow cannot be promoted as a recovered factual memory")
        retained = []
        for m in view["records"]:
            if m.state == "pending":
                continue
            if m.id == memory_id:
                m = replace(m, state="active", epsilon=0.0,
                            metadata={**m.metadata, "pinned": True, "evidence_note": evidence_note})
            retained.append(m)
        retained = bound_states(retained, self.config)
        version = self.store.commit(view["version"], 0, retained, build_graph(retained, self.config),
                                    {"action": "explicit_promotion", "id": memory_id},
                                    weight_artifact=view["weight_artifact"])
        return {"version": version, "id": memory_id}

    def endogenous(self, count=3):
        if not isinstance(count, int) or not 1 <= count <= 20:
            raise ValueError("count must be 1..20")
        view = self.store.view()
        nodes = [m for m in view["records"] if m.state == "active"]
        if len(nodes) < 2:
            return []
        rng = np.random.default_rng(self.config.seed + view["version"])
        proposals = []
        for _ in range(count):
            i, j = rng.choice(len(nodes), 2, replace=False)
            direction = nodes[i].vector + nodes[j].vector + rng.normal(0, 0.02, self.config.dimension)
            if np.linalg.norm(direction) <= 1e-12:
                continue
            result = self.retrieve(vector=direction, k=3, include_pending=False)
            proposals.append({"seed_ids": [nodes[i].id, nodes[j].id], "status": "unverified",
                              "omega": result["omega"], "candidate_ids": [h.memory.id for h in result["hits"]]})
        # Endogenous exploration never silently creates or promotes facts.
        return proposals

    def stats(self):
        view = self.store.view()
        counts = {s: sum(m.state == s for m in view["records"]) for s in ("pending", "active", "shadow")}
        text_bytes = sum(len((m.text or "").encode()) for m in view["records"])
        vector_bytes = sum(m.vector.nbytes for m in view["records"])
        archive = [m for m in view["records"] if m.state == "shadow"]
        effective_eps = [m.epsilon * 2 ** (-(view["seq"] - m.seq) / self.config.shadow_half_life_steps) for m in archive]
        files = [Path(self.store.path + suffix) for suffix in ("", "-wal", "-shm")]
        disk = sum(p.stat().st_size for p in files if p.is_file()) if self.store.path != ":memory:" else None
        return {"version": view["version"], "sequence": view["seq"], **counts, "edges": len(view["edges"]),
                "identity_depth_count_proxy": counts["active"] + math.log1p(counts["shadow"]),
                "text_bytes": text_bytes, "vector_bytes": vector_bytes, "database_allocated_bytes": disk,
                "mean_effective_epsilon": float(np.mean(effective_eps)) if effective_eps else 0.0,
                "consolidation_threshold": self.config.consolidation_threshold,
                "weight_artifact": view["weight_artifact"], "encoder": self.encoder.fingerprint,
                "shadow_mode": self.config.shadow_mode, "selector": self.config.selector}

    def export(self):
        view = self.store.view()
        return {"format": "llicams-memory-v1", "config": self.config.to_dict(),
                "encoder": self.encoder.fingerprint, "version": view["version"],
                "records": [m.data(include_vector=True) for m in view["records"]],
                "edges": view["edges"], "events": self.store.events(), "proposals": self.store.proposals()}

    def close(self):
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
