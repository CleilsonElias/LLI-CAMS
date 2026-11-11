"""Pure numerical definitions, separate from claims about their usefulness."""
import math
import numpy as np


def unit(vector, dimension=None):
    x = np.asarray(vector, dtype=np.float64)
    if x.ndim != 1 or (dimension is not None and len(x) != dimension):
        raise ValueError("Invalid vector shape")
    if not np.isfinite(x).all():
        raise ValueError("Vector must be finite")
    norm = float(np.linalg.norm(x))
    if norm <= 1e-12:
        raise ValueError("Zero vectors cannot represent a semantic direction")
    return (x / norm).astype(np.float32)


def probability_vector(p):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1 or len(p) < 2 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Expected a finite non-negative probability vector with K >= 2")
    if not np.isclose(p.sum(), 1.0, atol=1e-7, rtol=0):
        raise ValueError("Cluster probabilities must sum to one")
    return p


def relational_entropy(p):
    """Original off-diagonal score in bits; not normalized pairwise Shannon entropy.

    Equivalent O(K) form of -sum_{i != j} p_i p_j log2(p_i p_j).
    """
    p = probability_vector(p)
    nz = p > 0
    return float(-2 * np.sum(p[nz] * (1 - p[nz]) * np.log2(p[nz])))


def normalized_relational_entropy(p):
    p = probability_vector(p)
    return float(np.clip(relational_entropy(p) / (2 * math.log2(len(p))), 0, 1))


def pair_entropy(p):
    p = probability_vector(p)
    z = 1 - float(p @ p)
    if z <= 1e-12:
        return 0.0  # explicit convention for an undefined conditional distribution
    return relational_entropy(p) / z + math.log2(z)


def sigmoid(x):
    x = float(x)
    if not math.isfinite(x):
        raise ValueError("Sigmoid input must be finite")
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1 + exp_x)


def historical_baseline(total, count, prior, warmup):
    gamma = -math.expm1(-count / warmup)
    observed = total / count if count else prior
    return (1 - gamma) * prior + gamma * observed


def urgency(score, baseline, alpha):
    return sigmoid(alpha * (score - baseline))


def cluster_affinities(vector, prototypes, temperature):
    v = unit(vector)
    prototypes = np.asarray(prototypes, dtype=np.float64)
    similarities = np.clip(prototypes @ v, -1, 1)
    logits = temperature * similarities
    ex = np.exp(logits - logits.max())
    p = ex / ex.sum()
    anchor = float(max(0, similarities.max()))
    return p, anchor


def project_orthogonal(update, basis):
    """Frobenius projection onto the complement of the span of all basis matrices.

    Does not imply functional non-interference in a nonlinear model.
    """
    u = np.asarray(update, dtype=np.float64)
    if not np.isfinite(u).all():
        raise ValueError("Non-finite update")
    if not basis:
        return u.copy()
    matrices = [np.asarray(b, dtype=np.float64) for b in basis]
    if any(b.shape != u.shape or not np.isfinite(b).all() for b in matrices):
        raise ValueError("Basis shapes must match the update")
    b = np.stack([x.ravel() for x in matrices], axis=1)
    coefficients = np.linalg.lstsq(b, u.ravel(), rcond=None)[0]
    return (u.ravel() - b @ coefficients).reshape(u.shape)


def rank_compress(matrix, rank):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or rank < 1 or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite matrix and positive rank")
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    r = min(rank, len(s))
    b = u[:, :r] * np.sqrt(s[:r])
    a = np.sqrt(s[:r])[:, None] * vh[:r]
    residual = float(np.linalg.norm(matrix - b @ a))
    return b, a, residual


def fit_linear_lora(x, y, rank=2, steps=300, learning_rate=0.05, seed=42):
    """Train an actual low-rank update of an identity linear map using NumPy.

    Numerical replay demonstration, NOT transformer fine-tuning.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if not isinstance(rank, int) or rank < 1 or not isinstance(steps, int) or steps < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("rank/steps must be positive integers and learning_rate finite and positive")
    if x.ndim != 2 or y.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("X and Y must have equal finite N x D shapes")
    rng = np.random.default_rng(seed)
    a = rng.normal(scale=0.05, size=(rank, x.shape[1]))
    b = np.zeros((x.shape[1], rank))
    losses = []
    for _ in range(steps):
        error = x + (x @ a.T) @ b.T - y
        losses.append(float(np.mean(error**2)))
        grad = 2 * error / error.size
        grad_b = grad.T @ (x @ a.T)
        grad_a = (grad @ b).T @ x
        b -= learning_rate * grad_b
        a -= learning_rate * grad_a
    return b, a, losses
