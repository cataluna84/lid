from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from lid.bench.strategies.vectorized import VectorizedStrategy
from lid.bench.strategy import StrategyRegistry

warnings.filterwarnings("ignore", message="MatMul8bitLt.*cast from torch.bfloat16")


@StrategyRegistry.register("quantized_int8")
class QuantizedInt8Strategy(VectorizedStrategy):
    """Vectorized extraction with INT8 quantization via bitsandbytes."""

    def load_model(
        self,
        model_name: str,
        dtype: str,
        device: str = "cuda",
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto",
        )
        return model, tokenizer


@StrategyRegistry.register("quantized_int4")
class QuantizedInt4Strategy(VectorizedStrategy):
    """Vectorized extraction with INT4 (NF4) quantization via bitsandbytes."""

    def load_model(
        self,
        model_name: str,
        dtype: str,
        device: str = "cuda",
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            ),
            device_map="auto",
        )
        return model, tokenizer
