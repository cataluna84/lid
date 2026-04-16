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

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # [L, B, S, D]
        all_hidden = torch.stack(outputs.hidden_states)

        n_batch = len(prompts)
        last_idx = inputs["attention_mask"].sum(dim=1) - 1  # [B]
        batch_range = torch.arange(n_batch, device=model.device)

        # [n_layers, B, D]
        last_hidden = all_hidden[:, batch_range, last_idx, :]

        # Single batched lm_head: [n_layers, B, V]
        n_layers, _, d_model = last_hidden.shape
        flat = last_hidden.reshape(n_layers * n_batch, d_model)
        flat_logits = model.lm_head(flat) / temperature
        all_logits = flat_logits.reshape(n_layers, n_batch, -1)

        all_lp = torch.log_softmax(all_logits, dim=-1)  # [n_layers, B, V]

        # Gather log-probs for class token IDs: [n_layers, B, n_cls, max_toks]
        n_cls, max_toks = token_ids.shape
        gathered = all_lp[:, :, token_ids.view(-1)].reshape(
            n_layers, n_batch, n_cls, max_toks
        )
        masked = gathered * token_mask.unsqueeze(0).unsqueeze(0)

        scores = masked.sum(dim=-1)  # [L, B, C]
        norm_probs = torch.softmax(scores, dim=-1)  # [L, B, C]

        del outputs, inputs, all_hidden, flat, flat_logits
        return norm_probs
