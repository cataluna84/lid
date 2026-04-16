from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from lid.bench.strategies.vectorized import VectorizedStrategy
from lid.bench.strategy import StrategyRegistry


@StrategyRegistry.register("combined_flash_compiled")
class CombinedFlashCompiledStrategy(VectorizedStrategy):
    """Flash Attention 2 + torch.compile."""

    def load_model(
        self,
        model_name: str,
        dtype: str,
        device: str = "cuda",
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
        ).to(device)
        model = torch.compile(model, mode="reduce-overhead")
        return model, tokenizer


@StrategyRegistry.register("combined_flash_int8")
class CombinedFlashInt8Strategy(VectorizedStrategy):
    """Flash Attention 2 + INT8 quantization."""

    def load_model(
        self,
        model_name: str,
        dtype: str,
        device: str = "cuda",
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        return model, tokenizer
