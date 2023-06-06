import math
import torch
import torch.nn as nn
from transformers import RobertaModel


class CodeBertEncoder(nn.Module):
    def __init__(self, hidden_size: int = 512) -> None:
        super().__init__()
        self.codebert = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.linear = nn.Linear(768, hidden_size)
        self.act1 = nn.ReLU()

    def forward(self, x: torch.Tensor, lengths: torch.Tensor,
                src_key_padding_mask: torch.Tensor):
        # attention_mask: 1 if not mask; 0 if mask
        # last_hidden_state: B, L, H
        feature = self.codebert(input_ids=x, attention_mask=src_key_padding_mask)[0]
        # pooled = torch.mean(feature, dim=1)  # B, H
        clss = feature[:, 0, :]  # B, H

        return self.act1(self.linear(clss))
