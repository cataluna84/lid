from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from lid.bench.strategies.vectorized import VectorizedStrategy
from lid.bench.strategy import StrategyRegistry


@StrategyRegistry.register("flash_attn")
class FlashAttnStrategy(VectorizedStrategy):
    """Vectorized extraction with Flash Attention 2."""

    def load_model(
        self,
        model_name: str,
        dtype: str,
        device: str = "cuda",
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
        ).to(device)
        return model, tokenizer


@StrategyRegistry.register("sdpa")
class SDPAStrategy(VectorizedStrategy):
    """Vectorized extraction with PyTorch scaled dot-product attention."""

    def load_model(
        self,
        model_name: str,
        dtype: str,
        device: str = "cuda",
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",
        ).to(device)
        return model, tokenizer
