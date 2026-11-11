"""Deterministic lexical encoder and optional neural sentence encoder."""
import hashlib
import json
import re
import unicodedata
import numpy as np
from .math import unit


def words(text):
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9_]+", text)


class HashEncoder:
    """Signed hashing of words/bigrams. Lexical baseline, not semantic understanding."""
    def __init__(self, dimension=384, seed=42):
        self.dimension, self.seed = dimension, seed
        self.fingerprint = f"hash-blake2b-words-v1:{dimension}:{seed}"

    def encode(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string")
        tokens = words(text)
        if not tokens:
            tokens = [text.strip()]
        features = tokens + [a + "_" + b for a, b in zip(tokens, tokens[1:])]
        vector = np.zeros(self.dimension)
        for feature in features:
            digest = hashlib.blake2b(f"{self.seed}:{feature}".encode(), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "little") % self.dimension
            vector[index] += 1 if digest[8] & 1 else -1
        if not np.any(vector):
            vector[int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") % self.dimension] = 1
        return unit(vector)


class SentenceTransformerEncoder:
    def __init__(self, model_name, revision=None, device="cpu", local_files_only=True):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install llicams-reference[semantic] to use a neural encoder") from exc
        self.model = SentenceTransformer(model_name, revision=revision, device=device,
                                         local_files_only=local_files_only, trust_remote_code=False)
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        descriptor = json.dumps([model_name, revision, self.dimension])
        self.fingerprint = "sentence-transformers:" + hashlib.sha256(descriptor.encode()).hexdigest()

    def encode(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be non-empty")
        return unit(self.model.encode(text, normalize_embeddings=True), self.dimension)
