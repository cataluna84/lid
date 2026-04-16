from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lid.bench.strategy import InferenceStrategy, StrategyRegistry


@StrategyRegistry.register("eager")
class EagerStrategy(InferenceStrategy):
    """Baseline eager-mode inference matching the original notebook code."""

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
        option_token_ids = {
            opt: tokenizer.encode(opt, add_special_tokens=False) for opt in valid_options
        }
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        batch_size = len(prompts)
        total_layers = len(model.model.layers) + 1
        n_classes = len(valid_options)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        result = torch.zeros(total_layers, batch_size, n_classes, device=model.device)

        for b in range(batch_size):
            last_idx = (inputs["attention_mask"][b].sum() - 1).item()
            for layer_idx in range(total_layers):
                hidden = outputs.hidden_states[layer_idx][b, last_idx, :].unsqueeze(0)
                logits = model.lm_head(hidden) / temperature
                log_probs = torch.log_softmax(logits, dim=-1)

                scores = []
                for opt in valid_options:
                    toks = option_token_ids[opt]
                    score = sum(log_probs[0, t].item() for t in toks)
                    scores.append(score)

                lp = torch.tensor(scores, device=model.device)
                result[layer_idx, b] = torch.softmax(lp, dim=0)

        del outputs, inputs
        return result
