import numpy as np
from .math import unit


class VectorIndex:
    """Immutable exact/HNSW index. Build and queries never mutate the same object."""
    def __init__(self, records, kind="exact", seed=42):
        self.records = tuple(sorted(records, key=lambda m: m.id))
        self.kind = kind
        self.ef_search = 100
        self.matrix = np.stack([m.vector for m in self.records]) if self.records else None
        self.hnsw = None
        if kind == "hnsw" and self.records:
            try:
                import hnswlib
            except ImportError as exc:
                raise RuntimeError("Install llicams-reference[hnsw] or use index='exact'") from exc
            self.hnsw = hnswlib.Index(space="cosine", dim=self.matrix.shape[1])
            self.hnsw.init_index(max_elements=len(self.records), ef_construction=100,
                                 M=16, random_seed=seed)
            self.hnsw.add_items(self.matrix, np.arange(len(self.records)), num_threads=1)
            self.hnsw.set_ef(self.ef_search)

    def search(self, query, k=10):
        if k < 1 or not self.records:
            return []
        query = unit(query, self.matrix.shape[1])
        k = min(k, len(self.records))
        if self.hnsw is not None and k <= self.ef_search:
            indices, _ = self.hnsw.knn_query(query, k=k, num_threads=1)
            candidates = indices[0]
        else:
            candidates = np.argsort(-(self.matrix @ query), kind="stable")[:k]
        matches = [(self.records[int(i)], float(np.clip(self.matrix[int(i)] @ query, -1, 1)))
                   for i in candidates]
        return sorted(matches, key=lambda pair: (-pair[1], pair[0].id))
