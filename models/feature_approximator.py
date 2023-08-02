import torch
import torch.nn as nn
from .mlp import MLP1


class FeatureApproximator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def get_transform_embedding(self, transform_onehots: torch.Tensor):
        raise NotImplementedError()


class TransformerApproximator(FeatureApproximator):
    def __init__(self,
                 vocab_size: int,
                 transform_capacity: int,
                 input_dim: int,
                 output_dim: int,
                 dropout: float = 0.2,
                 n_heads: int = 4,
                 n_layers: int = 1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.transform_capacity = transform_capacity

        self.t_embeddings = nn.Embedding(transform_capacity, input_dim)

        transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim,
                                                       nhead=n_heads,
                                                       dropout=dropout,
                                                       batch_first=True,
                                                       dim_feedforward=768)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=n_layers)
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

    def get_transform_embedding(self, transform_onehots: torch.Tensor):
        return torch.matmul(transform_onehots, self.t_embeddings.weight)

    def forward(self, code_feature: torch.Tensor, var_embedding: torch.Tensor,
                transform_embedding: torch.Tensor, src_mask: torch.Tensor):
        # code_feature: B, L, H
        # embeddings: B, H
        var_embedding = var_embedding.unsqueeze(1)
        transform_embedding = transform_embedding.unsqueeze(1)
        cat_feature = torch.cat([var_embedding, transform_embedding, code_feature], dim=1)

        # B, S -> B, (S+2), make up for the newly appended embeddings
        src_mask = torch.cat(
            [torch.zeros(src_mask.shape[0], 2).to(src_mask.device).bool(), src_mask],
            dim=1)

        # B, L', H
        feature = self.transformer(cat_feature, src_key_padding_mask=src_mask)
        feature = torch.mean(feature, dim=1)  # B, H

        return self.dropout1(self.act1(self.fc1(feature)))


class WeightedSumApproximator(FeatureApproximator):
    def __init__(self,
                 vocab_size: int,
                 transform_capacity: int,
                 input_dim: int,
                 output_dim: int,
                 dropout: float = 0.2,
                 bn: bool = False) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.transform_capacity = transform_capacity

        self.t_embeddings = nn.Embedding(transform_capacity, input_dim)
        self.t_warper = MLP1(input_dim * 2, output_dim)
        self.v_warper = MLP1(input_dim * 2, output_dim)

        self.wt = 0.5
        self.wv = 0.5

    def get_transform_embedding(self, transform_onehots: torch.Tensor):
        return torch.matmul(transform_onehots, self.t_embeddings.weight)

    def forward(self, code_feature: torch.Tensor, var_embedding: torch.Tensor,
                transform_embedding: torch.Tensor, src_mask: torch.Tensor):
        assert code_feature.shape[1] == var_embedding.shape[1]
        assert code_feature.shape[1] == transform_embedding.shape[1]

        ev = self.v_warper(torch.cat([code_feature, var_embedding], dim=1))
        et = self.t_warper(torch.cat([code_feature, transform_embedding], dim=1))

        return self.wv * ev + self.wt * et


class AdditionApproximator(FeatureApproximator):
    def __init__(self,
                 vocab_size: int,
                 transform_capacity: int,
                 input_dim: int,
                 output_dim: int,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.transform_capacity = transform_capacity
        self.t_embeddings = nn.Embedding(transform_capacity, input_dim)

        self.fc1 = nn.Linear(input_dim, input_dim)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(input_dim, output_dim)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.out = nn.Linear(output_dim, output_dim)

    def get_transform_embedding(self, transform_onehots: torch.Tensor):
        return torch.matmul(transform_onehots, self.t_embeddings.weight)

    def forward(self, code_feature: torch.Tensor, var_embedding: torch.Tensor,
                transform_embedding: torch.Tensor, src_mask: torch.Tensor):
        assert code_feature.shape[1] == var_embedding.shape[1]
        assert code_feature.shape[1] == transform_embedding.shape[1]

        x = code_feature + var_embedding + transform_embedding
        x = self.dropout1(self.act1(self.fc1(x)))
        x = self.dropout2(self.act2(self.fc2(x)))
        x = self.out(x)

        return x


class ConcatApproximator(FeatureApproximator):
    def __init__(self,
                 vocab_size: int,
                 transform_capacity: int,
                 input_dim: int,
                 output_dim: int,
                 dropout: float = 0.2,
                 bn: bool = False) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.transform_capacity = transform_capacity

        self.t_embeddings = nn.Embedding(transform_capacity, input_dim)
        self.fc1 = nn.Linear(input_dim * 3, input_dim * 2)
        if bn:
            self.bn1 = nn.BatchNorm1d(input_dim * 2)
        else:
            self.bn1 = nn.Identity()
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(input_dim * 2, output_dim)
        if bn:
            self.bn2 = nn.BatchNorm1d(output_dim)
        else:
            self.bn2 = nn.Identity()
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

    def get_transform_embedding(self, transform_onehots: torch.Tensor):
        return torch.matmul(transform_onehots, self.t_embeddings.weight)

    def forward(self, code_feature: torch.Tensor, var_embedding: torch.Tensor,
                transform_embedding: torch.Tensor, src_mask: torch.Tensor):
        assert code_feature.shape[1] == var_embedding.shape[1]
        assert code_feature.shape[1] == transform_embedding.shape[1]

        x = torch.cat([code_feature, var_embedding, transform_embedding], dim=1)
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.act2(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return x


class VarApproximator(FeatureApproximator):
    def __init__(self,
                 vocab_size: int,
                 input_dim: int,
                 output_dim: int,
                 dropout: float = 0.2,
                 bn: bool = False) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.fc1 = nn.Linear(input_dim * 2, input_dim * 2)
        if bn:
            self.bn1 = nn.BatchNorm1d(input_dim * 2)
        else:
            self.bn1 = nn.Identity()
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(input_dim * 2, output_dim)
        if bn:
            self.bn2 = nn.BatchNorm1d(output_dim)
        else:
            self.bn2 = nn.Identity()
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

    def get_transform_embedding(self, transform_onehots: torch.Tensor):
        raise RuntimeError("VarApproximator does not support transform embedding")

    def forward(self, code_feature: torch.Tensor, var_embedding: torch.Tensor,
                src_mask: torch.Tensor):
        assert code_feature.shape[1] == var_embedding.shape[1]

        x = torch.cat([code_feature, var_embedding], dim=1)
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.act2(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return x


class TransformationApproximator(FeatureApproximator):
    def __init__(self,
                 transform_capacity: int,
                 input_dim: int,
                 output_dim: int,
                 dropout: float = 0.2,
                 bn: bool = False) -> None:
        super().__init__()
        self.transform_capacity = transform_capacity

        self.t_embeddings = nn.Embedding(transform_capacity, input_dim)
        self.fc1 = nn.Linear(input_dim * 2, input_dim * 2)
        if bn:
            self.bn1 = nn.BatchNorm1d(input_dim * 2)
        else:
            self.bn1 = nn.Identity()
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(input_dim * 2, output_dim)
        if bn:
            self.bn2 = nn.BatchNorm1d(output_dim)
        else:
            self.bn2 = nn.Identity()
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

    def get_transform_embedding(self, transform_onehots: torch.Tensor):
        return torch.matmul(transform_onehots, self.t_embeddings.weight)

    def forward(self, code_feature: torch.Tensor, transform_embedding: torch.Tensor,
                src_mask: torch.Tensor):
        assert code_feature.shape[1] == transform_embedding.shape[1]

        x = torch.cat([code_feature, transform_embedding], dim=1)
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.act2(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return x
