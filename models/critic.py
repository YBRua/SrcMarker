import torch
import torch.nn as nn


class DecodeLossApproximator(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 transform_capacity: int,
                 code_feature_dim: int,
                 embedding_dim: int,
                 output_dim: int,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.transform_capacity = transform_capacity
        self.t_embeddings = nn.Embedding(transform_capacity, embedding_dim)

        self.feature_align = nn.Linear(code_feature_dim, embedding_dim)
        self.feature_align_act = nn.ReLU()

        self.fc1 = nn.Linear(embedding_dim, embedding_dim * 2)
        # self.bn1 = nn.BatchNorm1d(embedding_dim * 2)
        self.bn1 = nn.Identity()
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(embedding_dim * 2, embedding_dim)
        # self.bn2 = nn.BatchNorm1d(embedding_dim)
        self.bn2 = nn.Identity()
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(embedding_dim, output_dim)

    def forward(self, code_feature: torch.Tensor, var_embedding: torch.Tensor,
                transform_embedding: torch.Tensor):
        # code_feature: B, H1
        # embedding: B, S, H2
        B, S, _ = var_embedding.shape

        # B, 1, H2
        code_feature = self.feature_align_act(
            self.feature_align(code_feature).unsqueeze(1))

        # B, S, H2
        x = code_feature + var_embedding + transform_embedding
        x = x.reshape(B * S, -1)
        x = self.dropout1(self.act1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.act2(self.bn2(self.fc2(x))))

        # B, S, n_bits
        x = self.fc3(x)
        x = x.reshape(B, S, -1)

        return x
