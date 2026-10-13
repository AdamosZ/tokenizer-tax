"""Tokenizer registry.

Every tokenizer is wrapped as a plain ``str -> int`` function returning the
number of tokens *without* special tokens (BOS/EOS), so the counts measure the
text only.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

# name shown in results -> (backend, identifier)
REGISTRY: dict[str, tuple[str, str]] = {
    "GPT-4 / 3.5 (cl100k)": ("tiktoken", "cl100k_base"),
    "GPT-4o (o200k)": ("tiktoken", "o200k_base"),
    "Llama 3": ("hf", "unsloth/Llama-3.2-1B"),
    "Mistral 7B v0.3": ("hf", "unsloth/mistral-7b-v0.3"),
    "Mistral Nemo (Tekken)": ("hf", "mistralai/Mistral-Nemo-Base-2407"),
    "Qwen 2.5": ("hf", "Qwen/Qwen2.5-0.5B"),
    "Gemma 3": ("hf", "unsloth/gemma-3-1b-pt"),
    "CroissantLLM (FR-EN)": ("hf", "croissantllm/CroissantLLMBase"),
}


@dataclass(frozen=True)
class Counter:
    name: str
    count: Callable[[str], int]


@lru_cache(maxsize=None)
def load(name: str) -> Counter:
    backend, ident = REGISTRY[name]
    if backend == "tiktoken":
        import tiktoken

        enc = tiktoken.get_encoding(ident)
        return Counter(name, lambda s: len(enc.encode(s, disallowed_special=())))
    if backend == "hf":
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(hf_hub_download(ident, "tokenizer.json"))
        return Counter(name, lambda s: len(tok.encode(s, add_special_tokens=False).ids))
    raise ValueError(f"unknown backend {backend!r}")


def load_all(names: list[str] | None = None) -> list[Counter]:
    return [load(n) for n in (names or list(REGISTRY))]
