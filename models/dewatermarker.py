from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer
import math

# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html


# helper Module that adds positional encoding to the token embedding
# to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


# helper Module to convert tensor of input indices
# into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        batch_first: bool = False,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.generator = nn.Linear(emb_size, vocab_size)
        self.tok_emb = TokenEmbedding(vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
    ):
        src_emb = self.positional_encoding(self.tok_emb(src))
        tgt_emb = self.positional_encoding(self.tok_emb(tgt))
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(
            self.positional_encoding(self.tok_emb(src)), src_mask
        )

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.tok_emb(tgt)), memory, tgt_mask
        )


def generate_square_subsequent_mask(sz: int, device: torch.device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src: Tensor, tgt: Tensor, pad_idx: int, device: torch.device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).bool()

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class Seq2SeqAttentionGRU(nn.Module):
    def __init__(
        self, vocab_size: int, hidden_size: int, bos_idx: int, dropout_p: float = 0.1
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.hidden_size = hidden_size
        self.encoder_gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.encoder_dropout = nn.Dropout(dropout_p)

        self.attention = BahdanauAttention(hidden_size)
        self.decoder_gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.decoder_dropout = nn.Dropout(dropout_p)
        self.bos_idx = bos_idx

    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor):
        device = input_ids.device
        batch_size = input_ids.size(0)

        if target_ids is not None:
            target_ids_input = target_ids[:, 1:]
            max_len = target_ids_input.size(1)
        else:
            target_ids_input = None
            max_len = 256

        embedded = self.encoder_dropout(self.embedding(input_ids))
        encoder_outputs, encoder_hidden = self.encoder_gru(embedded)

        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=device
        ).fill_(self.bos_idx)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(max_len):
            decoder_output, decoder_hidden, _ = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)

            if target_ids_input is not None:
                # Teacher forcing: Feed the target as the next input
                should_teacher_force = torch.rand(1).item() < 0.5
                if should_teacher_force:
                    decoder_input = target_ids_input[:, i].unsqueeze(
                        1
                    )  # Teacher forcing
                else:
                    _, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze(-1).detach()
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(
                    -1
                ).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        return decoder_outputs

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.decoder_dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.decoder_gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights
