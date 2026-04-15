import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lid.constants import DEFAULT_MODEL, VALID_OPTIONS


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
    dtype: torch.dtype | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype
    ).to(device)

    return model, tokenizer, device


def layer_text_outputs(
    model,
    tokenizer,
    prompt: str,
    valid_options: list[str] | None = None,
    max_new_tokens: int = 10,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_length: int = 512,
) -> dict:
    if valid_options is None:
        valid_options = VALID_OPTIONS

    option_token_ids = {
        opt: tokenizer.encode(opt, add_special_tokens=False) for opt in valid_options
    }
    max_option_tokens = max(len(v) for v in option_token_ids.values())

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_length
    ).to(model.device)
    input_ids = inputs["input_ids"]
    total_layers = len(model.model.layers) + 1

    start = time.time()
    log_probs = {opt: [0.0] * total_layers for opt in valid_options}
    running_ids = input_ids.clone()

    with torch.no_grad():
        for step in range(max_option_tokens):
            outputs = model(running_ids, output_hidden_states=True, use_cache=True)
            for layer_idx in range(total_layers):
                hidden = outputs.hidden_states[layer_idx]
                logits = model.lm_head(hidden)[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                for opt in valid_options:
                    toks = option_token_ids[opt]
                    if step < len(toks):
                        log_probs[opt][layer_idx] += torch.log(
                            probs[0, toks[step]] + 1e-40
                        ).item()

            sorted_probs, sorted_indices = torch.sort(
                outputs.logits[:, -1, :] / temperature, descending=True
            )
            sorted_probs = torch.softmax(sorted_probs, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_probs[mask] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_token_id = sorted_indices.gather(
                -1, torch.multinomial(sorted_probs, 1)
            )
            running_ids = torch.cat([running_ids, next_token_id], dim=1)

            del outputs, hidden, logits, probs, sorted_probs, sorted_indices
            del cumulative, mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    layer_outputs = {}
    for layer_idx in range(total_layers):
        log_vals = torch.tensor(
            [log_probs[opt][layer_idx] for opt in valid_options]
        )
        norm_vals = torch.softmax(log_vals, dim=0)
        layer_outputs[f"LAYER-{str(layer_idx + 1).zfill(3)}"] = {
            f"norm_prob_{opt}": norm_vals[i].item()
            for i, opt in enumerate(valid_options)
        }
    layer_outputs["runtime_seconds"] = time.time() - start
    return layer_outputs


def batched_layer_text_outputs(
    model,
    tokenizer,
    prompts: list[str],
    valid_options: list[str] | None = None,
    temperature: float = 0.2,
    max_length: int = 512,
) -> list[dict]:
    if valid_options is None:
        valid_options = VALID_OPTIONS

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

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    batch_results = []
    for b in range(batch_size):
        layer_data = {}
        last_token_idx = (inputs["attention_mask"][b].sum() - 1).item()

        for layer_idx in range(total_layers):
            hidden = outputs.hidden_states[layer_idx][
                b, last_token_idx, :
            ].unsqueeze(0)
            logits = model.lm_head(hidden) / temperature
            log_probs_t = torch.log_softmax(logits, dim=-1)

            row_probs = {}
            for opt in valid_options:
                toks = option_token_ids[opt]
                score = sum(log_probs_t[0, t].item() for t in toks)
                row_probs[f"norm_prob_{opt}"] = score

            lp_tensor = torch.tensor(
                [row_probs[f"norm_prob_{opt}"] for opt in valid_options]
            )
            norm_vals = torch.softmax(lp_tensor, dim=0)
            layer_data[f"LAYER-{str(layer_idx + 1).zfill(3)}"] = {
                f"norm_prob_{opt}": norm_vals[idx].item()
                for idx, opt in enumerate(valid_options)
            }

        batch_results.append(layer_data)

    del outputs, inputs
    return batch_results
