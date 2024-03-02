import torch
import torch.nn as nn


class WMEmbeddingEncoder(nn.Module):
    def __init__(self, n_embeddings: int, embedding_dim: int = 512) -> None:
        super().__init__()
        self.embd = nn.Embedding(n_embeddings, embedding_dim)

    def forward(self, x: torch.LongTensor):
        return self.embd(x)


class WMLinearEncoder(nn.Module):
    def __init__(self, n_bits: int, embedding_dim: int = 512) -> None:
        super().__init__()
        self.linear = nn.Linear(n_bits, embedding_dim)

    def forward(self, x: torch.Tensor):
        return self.linear(x)


class WMConcatEncoder(nn.Module):
    def __init__(
        self, n_bits: int, embedding_dim: int = 512, output_dim: int = 512
    ) -> None:
        super().__init__()
        self.n_bits = n_bits
        self.embedding = WMEmbeddingEncoder(2, embedding_dim)
        self.linear = nn.Linear(self.n_bits * embedding_dim, output_dim)

    def forward(self, x: torch.Tensor):
        # x: B, N
        # embedding: B, N, H
        embedding = self.embedding(x.long())
        # B, NH
        embedding = embedding.reshape(embedding.shape[0], -1)
        return self.linear(embedding)
