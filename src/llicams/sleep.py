"""Four-phase planner. Missing optional replay is reported as skipped, never simulated."""
from dataclasses import replace
import hashlib
import time
import numpy as np


def priority(record, config):
    pinned = bool(record.metadata.get("pinned"))
    if config.selector == "recency":
        value = record.seq
    elif config.selector == "random":
        value = int.from_bytes(hashlib.sha256(f"{config.seed}:{record.id}".encode()).digest()[:8], "little")
    else:
        value = record.selection_score * record.confidence
    return (pinned, value, record.seq, record.id)


def bound_states(records, config, strip_shadow_text=True):
    active = sorted((m for m in records if m.state == "active"), key=lambda m: priority(m, config), reverse=True)
    shadows = [m for m in records if m.state == "shadow"]
    pinned = sum(bool(m.metadata.get("pinned")) for m in active)
    if pinned > config.max_active:
        from .storage import CapacityError
        raise CapacityError("Pinned memories exceed active capacity")
    shadows += [replace(m, state="shadow", epsilon=config.epsilon_max / 2) for m in active[config.max_active:]]
    current_seq = max((m.seq for m in records), default=0)
    def shadow_priority(m):
        p = priority(m, config)
        if config.selector in {"recency", "random"}:
            return p
        effective = m.epsilon * 2 ** (-(current_seq - m.seq) / config.shadow_half_life_steps)
        return (p[0], p[1] * effective, p[2], p[3])
    shadows.sort(key=shadow_priority, reverse=True)
    shadows = shadows[:config.max_shadows]
    if config.shadow_mode == "direction_only" and strip_shadow_text:
        shadows = [replace(m, text=None, metadata={"pinned": False}) for m in shadows]
    return active[:config.max_active] + shadows


def build_graph(records, config, penalties=None):
    nodes = sorted((m for m in records if m.state == "active"), key=lambda m: m.id)
    candidates = []
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            weight = max(0.0, float(a.vector @ b.vector))
            if penalties:
                weight *= penalties.get((a.id, b.id), 1.0)
            if weight >= config.graph_min_similarity and weight > 0:
                candidates.append((a.id, b.id, weight))
    degrees, edges = {}, []
    for a, b, weight in sorted(candidates, key=lambda x: (-x[2], x[0], x[1])):
        if degrees.get(a, 0) < config.max_degree and degrees.get(b, 0) < config.max_degree:
            edges.append((a, b, weight))
            degrees[a] = degrees.get(a, 0) + 1
            degrees[b] = degrees.get(b, 0) + 1
    return edges


def conflicting(a, b):
    """Only explicit supplied fact_key/fact_value contradictions, not truth detection."""
    ka, kb = a.metadata.get("fact_key"), b.metadata.get("fact_key")
    return bool(ka and ka == kb and a.metadata.get("fact_value") != b.metadata.get("fact_value"))


def plan(view, config, replay=None, rem=None):
    start = time.perf_counter()
    pending = [m for m in view["records"] if m.state == "pending"]
    records = [m for m in view["records"] if m.state != "pending"]
    phases = []
    # I: similar pending items form deterministic connected components.
    groups, visited = [], set()
    for root in pending:
        if root.id in visited:
            continue
        group, stack = [], [root]
        visited.add(root.id)
        while stack:
            item = stack.pop()
            group.append(item)
            for other in pending:
                if other.id not in visited and float(item.vector @ other.vector) >= config.cluster_similarity:
                    visited.add(other.id)
                    stack.append(other)
        groups.append(group)
    for group in groups:
        representative = max(group, key=lambda m: priority(m, config))
        for m in group:
            forced = bool(m.metadata.get("pinned"))
            eligible = m.confidence >= config.min_confidence
            is_active = forced or (eligible and m.id == representative.id and
                                  (config.selector in {"recency", "random"} or
                                   m.selection_score >= config.consolidation_threshold))
            if is_active:
                records.append(replace(m, state="active", epsilon=0.0))
            elif eligible and m.selection_score >= config.shadow_threshold:
                eps = config.epsilon_max * min(0.99, max(0.01, m.selection_score))
                records.append(replace(m, state="shadow", epsilon=eps))
    records = bound_states(records, config, strip_shadow_text=False)
    phases.append({"phase": "I_SWS", "status": "completed", "pending": len(pending), "groups": len(groups)})
    # II: concrete optional hook can train a transformer adapter; preparation must not mutate the live model.
    artifact = view["weight_artifact"]
    replay_report = {"phase": "II_REPLAY", "status": "skipped", "reason": "no replay backend configured"}
    if replay is not None:
        candidates = [m for m in records if m.state == "active" and m.text is not None]
        result = replay.prepare(candidates, artifact)
        artifact = result["artifact"]
        replay_report = {"phase": "II_REPLAY", "status": "prepared", **result.get("metrics", {})}
    phases.append(replay_report)
    # III: bounded cross-cluster proposals; explicit contradictory facts weaken edges.
    active = [m for m in records if m.state == "active"]
    pairs = [(a, b) for i, a in enumerate(active) for b in active[i + 1:]
             if conflicting(a, b) or int(np.argmax(a.probabilities)) != int(np.argmax(b.probabilities))]
    pairs.sort(key=lambda pair: hashlib.sha256(f"{config.seed}:{pair[0].id}:{pair[1].id}".encode()).digest())
    penalties, proposals = {}, []
    for a, b in pairs[:config.rem_pairs]:
        contradiction = conflicting(a, b)
        pair = tuple(sorted((a.id, b.id)))
        if contradiction:
            penalties[pair] = config.rem_attenuation
        if rem is not None:
            proposal = rem.propose(a, b)
            if proposal.get("coherence", 1.0) < 0.5:
                penalties[pair] = config.rem_attenuation
        else:
            proposal = {"hypothesis": "Avaliar uma possível relação entre as duas memórias; nenhuma relação factual foi inferida.",
                        "critic": "explicit_metadata_only", "status": "unverified"}
        proposals.append({**proposal, "sources": [a.id, b.id], "explicit_conflict": contradiction,
                          "status": "unverified"})
    phases.append({"phase": "III_REM", "status": "completed", "mode": "llm" if rem else "structural",
                   "proposals": len(proposals), "truth_guarantee": False})
    # IV: archival and limits. direction_only strips text AND metadata payload.
    records = bound_states(records, config)
    edges = build_graph(records, config, penalties)
    phases.append({"phase": "IV_ARCHIVE", "status": "completed", "mode": config.shadow_mode,
                   "active": sum(m.state == "active" for m in records),
                   "shadows": sum(m.state == "shadow" for m in records)})
    report = {"base_version": view["version"], "cutoff": view["seq"], "phases": phases,
              "duration_seconds": time.perf_counter() - start,
              "dropped": len(view["records"]) - len(records)}
    return records, edges, proposals, artifact, report
