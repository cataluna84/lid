from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class InferenceStrategy(ABC):
    """Abstract base class for inference optimization strategies."""

    name: str = "base"

    @abstractmethod
    def load_model(
        self,
        model_name: str,
        dtype: str,
        device: str = "cuda",
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Load model and tokenizer with strategy-specific configuration."""

    @abstractmethod
    def extract_layer_probs(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        prompts: list[str],
        valid_options: list[str],
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        temperature: float = 0.2,
        max_length: int = 512,
    ) -> torch.Tensor:
        """Extract per-layer class probabilities for a batch.

        Returns:
            Tensor of shape ``[n_layers, batch_size, n_classes]`` with
            normalized probabilities.
        """


class StrategyRegistry:
    """Registry mapping strategy names to their implementations."""

    _strategies: ClassVar[dict[str, type[InferenceStrategy]]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a strategy class."""

        def wrapper(strategy_cls: type[InferenceStrategy]):
            strategy_cls.name = name
            cls._strategies[name] = strategy_cls
            return strategy_cls

        return wrapper

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> InferenceStrategy:
        if name not in cls._strategies:
            available = ", ".join(sorted(cls._strategies.keys()))
            raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
        return cls._strategies[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._strategies.keys())


def build_token_index(
    tokenizer: PreTrainedTokenizerBase,
    valid_options: list[str],
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-build token ID index tensors for all class labels.

    Returns:
        token_ids: ``[n_classes, max_tok_len]`` padded token IDs.
        token_mask: ``[n_classes, max_tok_len]`` float mask (1.0 for valid).
    """
    all_tids = [tokenizer.encode(opt, add_special_tokens=False) for opt in valid_options]
    max_tok_len = max(len(t) for t in all_tids)
    n_classes = len(valid_options)

    token_ids = torch.zeros(n_classes, max_tok_len, dtype=torch.long, device=device)
    token_mask = torch.zeros(n_classes, max_tok_len, device=device)

    for i, tids in enumerate(all_tids):
        token_ids[i, : len(tids)] = torch.tensor(tids, device=device)
        token_mask[i, : len(tids)] = 1.0

    return token_ids, token_mask
