import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoderExtractor(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_size: int = 512,
                 hidden_size: int = 512,
                 num_layers: int = 3,
                 n_heads: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout_rate = dropout
        self.embedding_size = embedding_size
        self.n_heads = n_heads

        self.positional_encoding = PositionalEncoding(embedding_size, dropout)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size,
                                                   nhead=n_heads,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.code_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc1 = nn.Linear(embedding_size, hidden_size)
        self.act1 = nn.ReLU()

    def forward(self, x: torch.Tensor, lengths: torch.Tensor,
                src_key_padding_mask: torch.Tensor):
        # B, L, H
        embedding = self.embedding(x) * math.sqrt(self.embedding_size)
        embedding = self.positional_encoding(embedding)

        # B, L, H
        feature = self.code_encoder(embedding, src_key_padding_mask=src_key_padding_mask)
        pooled = torch.mean(feature, dim=1)  # B, H

        pooled = self.fc1(pooled)
        pooled = self.act1(pooled)

        return pooled
