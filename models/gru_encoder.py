import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class ExtractGRUEncoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_size: int = 512,
                 hidden_size: int = 512,
                 num_layers: int = 2,
                 bidirectional: bool = True,
                 dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.rnn_unit = nn.GRU(input_size=embedding_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               bidirectional=True)

        D = 2 if bidirectional else 1
        rnn_out_size = D * num_layers * hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(rnn_out_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor,
                src_key_padding_mask: torch.Tensor):
        B = x.shape[0]
        x = self.embedding(x)
        x = self.embedding_dropout(x)

        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn_unit(x)
        hidden = hidden.permute(1, 0, 2).reshape(B, -1)

        feature = self.mlp(hidden)

        return feature


class GRUEncoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_size: int = 512,
                 hidden_size: int = 512,
                 num_layers: int = 2,
                 bidirectional: bool = True,
                 dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.rnn_unit = nn.GRU(input_size=embedding_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               bidirectional=True)

        D = 2 if bidirectional else 1
        rnn_out_size = D * num_layers * hidden_size

        self.fc1 = nn.Linear(rnn_out_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.act1 = nn.ReLU()

    def forward(self, x: torch.Tensor, lengths: torch.Tensor,
                src_key_padding_mask: torch.Tensor):
        B = x.shape[0]
        x = self.embedding(x)
        x = self.embedding_dropout(x)

        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn_unit(x)
        hidden = hidden.permute(1, 0, 2).reshape(B, -1)

        feature = self.act1(self.bn1(self.fc1(hidden)))

        return feature
