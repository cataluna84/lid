from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lid.bench.strategy import InferenceStrategy, StrategyRegistry


@StrategyRegistry.register("vectorized")
class VectorizedStrategy(InferenceStrategy):
    """Fully vectorized layer-wise extraction with zero Python loops."""

    def load_model(
        self,
        model_name: str,
        dtype: str,
        device: str = "cuda",
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype
        ).to(device)
        return model, tokenizer

    def extract_layer_probs(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompts: list[str],
        valid_options: list[str],
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        temperature: float = 0.2,
        max_length: int = 512,
    ) -> torch.Tensor:
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        n_batch = len(prompts)
        last_idx = inputs["attention_mask"].sum(dim=1) - 1  # [B]
        batch_range = torch.arange(n_batch, device=model.device)

        with torch.no_grad():
            # Use base model to skip computing full [B, S, V] logits (~8 GiB for bs=32/ml=512)
            base_outputs = model.model(**inputs, output_hidden_states=True)

        hidden_states = base_outputs.hidden_states
        del base_outputs
        last_hidden = torch.stack(
            [h[batch_range, last_idx, :] for h in hidden_states]
        )
        del hidden_states, inputs

        # Single batched lm_head: [n_layers, B, V]
        n_layers, _, d_model = last_hidden.shape
        flat = last_hidden.reshape(n_layers * n_batch, d_model)
        del last_hidden
        flat_logits = model.lm_head(flat) / temperature
        all_logits = flat_logits.reshape(n_layers, n_batch, -1)
        del flat, flat_logits

        all_lp = torch.log_softmax(all_logits, dim=-1)  # [n_layers, B, V]
        del all_logits

        # Gather log-probs for class token IDs: [n_layers, B, n_cls, max_toks]
        n_cls, max_toks = token_ids.shape
        gathered = all_lp[:, :, token_ids.view(-1)].reshape(
            n_layers, n_batch, n_cls, max_toks
        )
        del all_lp
        masked = gathered * token_mask.unsqueeze(0).unsqueeze(0)
        del gathered

        scores = masked.sum(dim=-1)  # [L, B, C]
        del masked
        norm_probs = torch.softmax(scores, dim=-1)  # [L, B, C]

        return norm_probs
