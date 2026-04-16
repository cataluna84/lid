from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from lid.bench.strategies.vectorized import VectorizedStrategy
from lid.bench.strategy import StrategyRegistry

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer


@StrategyRegistry.register("compiled")
class CompiledStrategy(VectorizedStrategy):
    """Vectorized extraction with ``torch.compile`` applied to the model."""

    def load_model(
        self,
        model_name: str,
        dtype: str,
        device: str = "cuda",
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        model, tokenizer = super().load_model(model_name, dtype, device)
        model = torch.compile(model, mode="reduce-overhead")
        return model, tokenizer
