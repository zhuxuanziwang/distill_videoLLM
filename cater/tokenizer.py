import hashlib
import importlib
import re
from dataclasses import dataclass

import torch

_TOKEN_RE = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class TokenizerConfig:
    vocab_size: int = 8192
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2


class HashTokenizer:
    def __init__(self, config: TokenizerConfig | None = None):
        self.config = config or TokenizerConfig()

    def _token_to_id(self, token: str) -> int:
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        value = int.from_bytes(digest[:4], "big")
        return 3 + value % (self.config.vocab_size - 3)

    def encode(self, text: str, max_len: int) -> torch.Tensor:
        tokens = _TOKEN_RE.findall(text.lower())
        ids = [self.config.bos_id]
        ids.extend(self._token_to_id(token) for token in tokens[: max_len - 2])
        ids.append(self.config.eos_id)
        if len(ids) < max_len:
            ids.extend([self.config.pad_id] * (max_len - len(ids)))
        return torch.tensor(ids[:max_len], dtype=torch.long)


class CLIPTokenizerWrapper:
    def __init__(self, model_id: str = "openai/clip-vit-base-patch16", local_files_only: bool = False):
        try:
            transformers = importlib.import_module("transformers")
        except ModuleNotFoundError as e:
            raise RuntimeError("transformers is required for CLIP tokenizer. Please install requirements.txt.") from e

        auto_tokenizer = getattr(transformers, "AutoTokenizer")
        self.tokenizer = auto_tokenizer.from_pretrained(model_id, use_fast=True, local_files_only=local_files_only)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.bos_token

    def encode(self, text: str, max_len: int) -> torch.Tensor:
        out = self.tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return out["input_ids"][0].to(torch.long)
