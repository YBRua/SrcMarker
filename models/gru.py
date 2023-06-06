import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class GRUClassifier(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 num_classes: int,
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
        self.act1 = nn.ReLU()
        self.fc_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        B = x.shape[0]
        x = self.embedding(x)
        x = self.embedding_dropout(x)

        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn_unit(x)
        hidden = hidden.permute(1, 0, 2).reshape(B, -1)

        feature = self.act1(self.fc1(hidden))
        output = self.fc2(self.fc_dropout(feature))

        return output, feature
