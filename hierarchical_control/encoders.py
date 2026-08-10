from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence

import torch
import torch.nn.functional as F


class TextEncoder(Protocol):
    dimension: int
    name: str
    mock_only: bool

    def encode(self, texts: Sequence[str]) -> torch.Tensor: ...


class HashingEncoder:
    """Frozen deterministic smoke-test encoder; never use for formal experiments."""

    mock_only = True

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension
        self.name = f"hashing-{dimension}"

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        rows = torch.zeros((len(texts), self.dimension), dtype=torch.float32)
        for row, text in enumerate(texts):
            tokens = re.findall(r"\w+|[^\w\s]", text.casefold(), flags=re.UNICODE)
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:8], "little") % self.dimension
                sign = 1.0 if digest[8] & 1 else -1.0
                rows[row, index] += sign
        return F.normalize(rows, p=2, dim=-1)


class MiniLMEncoder:
    """Frozen all-MiniLM-L6-v2 encoder for formal training."""

    mock_only = False

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the project with the 'minilm' extra") from exc
        self.model = SentenceTransformer(model_name, device=device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        self.name = model_name

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        with torch.no_grad():
            embeddings = self.model.encode(
                list(texts), convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False
            )
        return embeddings.detach().to(dtype=torch.float32, device="cpu")
