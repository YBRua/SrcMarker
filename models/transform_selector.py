import torch
import torch.nn as nn
from typing import Optional, List


class Selector(nn.Module):
    def __init__(self,
                 num_warpers: int,
                 input_dim: int = 512,
                 dropout: float = 0.2,
                 bn: bool = True) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim * 2, input_dim)
        if bn:
            self.bn = nn.BatchNorm1d(input_dim)
        else:
            self.bn = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

        self.selector = nn.Linear(input_dim, num_warpers)

    def forward(self, code_feature: torch.Tensor, wm_feature: torch.Tensor):
        assert code_feature.shape[1] == wm_feature.shape[1]
        x = torch.cat([code_feature, wm_feature], dim=1)
        x = self.fc(x)
        x = self.bn(x)
        x = self.dropout(self.act(x))
        x = self.selector(x)

        return x


class TransformSelector(nn.Module):
    vocab_mask: Optional[torch.Tensor]

    def __init__(self,
                 vocab_size: int,
                 transform_capacity: int,
                 input_dim: int = 512,
                 random_mask_prob: float = 0.5,
                 dropout: float = 0.2,
                 bn: bool = True,
                 vocab_mask: Optional[List[bool]] = None) -> None:
        super().__init__()
        self.mask_prob = random_mask_prob
        self.var_selector = Selector(vocab_size, input_dim, dropout, bn)
        self.transform_selector = Selector(transform_capacity, input_dim, dropout, bn)

        if vocab_mask is not None:
            self.register_buffer('vocab_mask', torch.tensor(vocab_mask, dtype=torch.bool))
        else:
            self.vocab_mask = None

    def _get_random_mask_like(self, like: torch.Tensor):
        return torch.rand_like(like) < self.mask_prob

    def var_selector_forward(self,
                             code_feature: torch.Tensor,
                             wm_feature: torch.Tensor,
                             variable_mask: Optional[torch.Tensor] = None,
                             random_mask: bool = True,
                             return_probs: bool = False):
        outputs = self.var_selector(code_feature, wm_feature)

        if variable_mask is None:
            variable_mask = self.vocab_mask

        if variable_mask is not None:
            outputs = torch.masked_fill(outputs, variable_mask, float('-inf'))

        if random_mask:
            rand_mask = self._get_random_mask_like(outputs)
            outputs = torch.masked_fill(outputs, rand_mask.bool(), float('-inf'))

        if return_probs:
            return torch.softmax(outputs, dim=-1)
        else:
            return outputs

    def transform_selector_forward(self,
                                   code_feature: torch.Tensor,
                                   wm_feature: torch.Tensor,
                                   transform_mask: Optional[torch.Tensor] = None,
                                   random_mask: bool = False,
                                   return_probs: bool = False):
        outputs = self.transform_selector(code_feature, wm_feature)

        if transform_mask is not None:
            outputs = torch.masked_fill(outputs, transform_mask, float('-inf'))

        if random_mask:
            rand_mask = self._get_random_mask_like(outputs)
            outputs = torch.masked_fill(outputs, rand_mask.bool(), float('-inf'))

        if return_probs:
            return torch.softmax(outputs, dim=-1)
        else:
            return outputs
