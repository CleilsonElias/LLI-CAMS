"""Optional real transformer LoRA replay. Imports are lazy; never required by the core."""
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import time
import numpy as np
from .math import project_orthogonal, rank_compress


class HFLoRAReplay:
    """Train a candidate on approved text, fuse with the previous adapter, then stage it.

    Orthogonal projection is in Frobenius geometry, layer by layer, against the
    cumulative previous update. Recompression to fixed rank can lose orthogonality.
    The manifest reports this error instead of promising preservation of old tasks.
    """
    def __init__(self, model_name, output_directory, *, revision=None, rank=8, steps=20,
                 learning_rate=5e-4, max_length=256, seed=42, device="cpu",
                 target_modules=("q_proj", "v_proj"), local_files_only=True):
        if rank < 1 or steps < 1 or max_length < 4 or learning_rate <= 0:
            raise ValueError("Invalid replay hyperparameters")
        self.model_name, self.revision = model_name, revision
        self.output_directory = Path(output_directory)
        self.rank, self.steps, self.learning_rate = rank, steps, learning_rate
        self.max_length, self.seed, self.device = max_length, seed, device
        self.target_modules, self.local_files_only = list(target_modules), local_files_only

    def prepare(self, records, previous_artifact):
        # Approval is supplied by the caller, not inferred from high H_R or model salience.
        texts = [m.text for m in records if m.text and m.metadata.get("replay_approved") is True]
        if not texts:
            return {"artifact": previous_artifact,
                    "metrics": {"status": "skipped", "reason": "no replay_approved memories", "examples": 0}}
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
            from safetensors.numpy import load_file
        except ImportError as exc:
            raise RuntimeError("Install llicams-reference[transformers] to run actual LoRA replay") from exc
        torch.manual_seed(self.seed)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision,
            local_files_only=self.local_files_only, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer needs an existing pad/eos token")
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(self.model_name, revision=self.revision,
            local_files_only=self.local_files_only, trust_remote_code=False, torch_dtype=torch.float32)
        lora_config = LoraConfig(r=self.rank, lora_alpha=self.rank, lora_dropout=0.0,
                                 target_modules=self.target_modules, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(base, lora_config).to(self.device)
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=self.learning_rate)
        encoded = [tokenizer(t, return_tensors="pt", truncation=True, max_length=self.max_length) for t in texts]
        encoded = [e for e in encoded if e.input_ids.shape[1] >= 2]
        if not encoded:
            raise ValueError("No replay example contains at least two tokens")
        rng, losses = np.random.default_rng(self.seed), []
        for _ in range(self.steps):
            batch = encoded[int(rng.integers(len(encoded)))].to(self.device)
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch, labels=batch.input_ids.clone())
            if not torch.isfinite(output.loss):
                raise RuntimeError("Non-finite replay loss; candidate not published")
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(output.loss.detach().cpu()))
        state = get_peft_model_state_dict(model)
        previous = {}
        if previous_artifact:
            manifest = json.loads((Path(previous_artifact) / "replay_manifest.json").read_text())
            if manifest["base_model"] != self.model_name or manifest.get("revision") != self.revision:
                raise ValueError("Previous adapter belongs to a different base model/revision")
            previous = load_file(str(Path(previous_artifact) / "adapter_model.safetensors"))
        projections = []
        for a_key in [k for k in state if "lora_A" in k and k.endswith("weight")]:
            b_key = a_key.replace("lora_A", "lora_B")
            a = state[a_key].detach().float().cpu().numpy()
            b = state[b_key].detach().float().cpu().numpy()
            update = b @ a  # alpha/r == 1 in this implementation
            if a_key in previous:
                old = previous[b_key] @ previous[a_key]
                projected = project_orthogonal(update, [old])
                merged = old + projected
                fused_b, fused_a, residual = rank_compress(merged, self.rank)
                actual_increment = fused_b @ fused_a - old
                denom = max(float(np.linalg.norm(old) * np.linalg.norm(actual_increment)), 1e-12)
                leakage = abs(float(np.sum(old * actual_increment))) / denom
                # Pad factors when a small model layer has dimension < requested rank.
                padded_a, padded_b = np.zeros_like(a), np.zeros_like(b)
                padded_a[:fused_a.shape[0]] = fused_a
                padded_b[:, :fused_b.shape[1]] = fused_b
                state[a_key] = torch.as_tensor(padded_a, dtype=state[a_key].dtype)
                state[b_key] = torch.as_tensor(padded_b, dtype=state[b_key].dtype)
                projections.append({"module": a_key, "recompression_residual": residual,
                                    "post_compression_relative_overlap": leakage})
        set_peft_model_state_dict(model, state)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix="adapter-", dir=self.output_directory)).resolve()
        model.save_pretrained(stage, safe_serialization=True)
        metrics = {"status": "prepared", "examples": len(encoded), "steps": self.steps,
                   "first_loss": losses[0], "last_loss": losses[-1], "projection": projections,
                   "training_means_verified_truth": False}
        manifest = {"format": "llicams-hf-replay-v1", "base_model": self.model_name,
                    "revision": self.revision, "seed": self.seed, "created_at": time.time(),
                    "dataset_sha256": hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest(),
                    "adapter_sha256": hashlib.sha256((stage / "adapter_model.safetensors").read_bytes()).hexdigest(),
                    "metrics": metrics}
        (stage / "replay_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        del optimizer, model, base
        gc.collect()
        return {"artifact": str(stage), "metrics": metrics}
