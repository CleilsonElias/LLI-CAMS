"""A real numerical fit of a low-rank linear update; not an LLM experiment."""
import json
import numpy as np
from llicams.math import fit_linear_lora, project_orthogonal, rank_compress

rng = np.random.default_rng(5)
x = rng.normal(size=(80, 4))
desired_update = np.diag([0.4, -0.3, 0.0, 0.0])
y = x + x @ desired_update.T
b, a, losses = fit_linear_lora(x, y, rank=2, steps=1500, learning_rate=0.3)
new = rng.normal(size=(4, 4))
projected = project_orthogonal(new, [b @ a])
fused_b, fused_a, residual = rank_compress(b @ a + projected, 2)
print(json.dumps({"experiment": "linear_numpy_lora_not_transformer", "first_loss": losses[0],
                  "last_loss": losses[-1], "orthogonal_inner_product": float(np.sum(projected * (b @ a))),
                  "recompression_residual": residual}, indent=2))
