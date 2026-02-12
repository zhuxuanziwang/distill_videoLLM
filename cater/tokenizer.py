import hashlib
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
