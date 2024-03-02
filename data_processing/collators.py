import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, Set


class TokenIdCollator:
    PAD_ID = 0

    def __call__(self, batch):
        token_ids = []
        lengths = []

        for token_id in batch:
            token_ids.append(torch.tensor(token_id, dtype=torch.long))
            lengths.append(len(token_id))

        padded = pad_sequence(token_ids, batch_first=True, padding_value=self.PAD_ID)
        mask = padded == self.PAD_ID

        return (
            pad_sequence(token_ids, batch_first=True, padding_value=self.PAD_ID),
            torch.tensor(lengths, dtype=torch.long),
            mask,
        )


class TokenIdCollatorCodeBert:
    def __init__(self, pad_idx: int) -> None:
        self.pad_idx = pad_idx

    def __call__(self, batch):
        token_ids = []
        lengths = []

        for token_id in batch:
            token_ids.append(torch.tensor(token_id, dtype=torch.long))
            lengths.append(len(token_id))

        padded = pad_sequence(token_ids, batch_first=True, padding_value=self.pad_idx)
        mask = (padded != self.pad_idx).long()  # NOTE: 1 if not mask; 0 if mask

        return (
            pad_sequence(token_ids, batch_first=True, padding_value=self.pad_idx),
            torch.tensor(lengths, dtype=torch.long),
            mask,
        )


class DynamicWMCollatorCodeBert:
    def __init__(self, n_bits: int, pad_idx: int) -> None:
        self.n_bits = n_bits
        self.pad_idx = pad_idx

    def __call__(self, batch):
        instance_ids = []
        token_ids = []
        lengths = []
        watermarks = []
        wmids = []
        for instance_id, token_id in batch:
            token_ids.append(torch.tensor(token_id, dtype=torch.long))
            lengths.append(len(token_id))
            instance_ids.append(instance_id)
            wm = torch.randint(0, 2, (self.n_bits,))
            watermarks.append(wm)
            wmids.append(sum([2**i * wm[i] for i in range(self.n_bits)]))

        padded = pad_sequence(token_ids, batch_first=True, padding_value=self.pad_idx)
        mask = (padded != self.pad_idx).long()  # NOTE: 1 if not mask; 0 if mask

        return (
            padded,
            torch.tensor(lengths, dtype=torch.long),
            mask,
            instance_ids,
            torch.stack(watermarks, dim=0),
            torch.tensor(wmids, dtype=torch.long),
        )


class DynamicWMCollator:
    PAD_ID = 0

    def __init__(self, n_bits: int) -> None:
        self.n_bits = n_bits

    def __call__(self, batch):
        instance_ids = []
        token_ids = []
        lengths = []
        watermarks = []
        wmids = []
        for instance_id, token_id in batch:
            token_ids.append(torch.tensor(token_id, dtype=torch.long))
            lengths.append(len(token_id))
            instance_ids.append(instance_id)
            wm = torch.randint(0, 2, (self.n_bits,))
            watermarks.append(wm)
            wmids.append(sum([2**i * wm[i] for i in range(self.n_bits)]))

        padded = pad_sequence(token_ids, batch_first=True, padding_value=self.PAD_ID)
        mask = padded == self.PAD_ID

        return (
            padded,
            torch.tensor(lengths, dtype=torch.long),
            mask,
            instance_ids,
            torch.stack(watermarks, dim=0),
            torch.tensor(wmids, dtype=torch.long),
        )


class TaskedDynamicWMCollator:
    PAD_ID = 0

    def __init__(self, n_bits: int) -> None:
        self.n_bits = n_bits

    def __call__(self, batch):
        instance_ids = []
        token_ids = []
        lengths = []
        watermarks = []
        wmids = []
        labels = []
        for instance_id, token_id, label in batch:
            token_ids.append(torch.tensor(token_id, dtype=torch.long))
            lengths.append(len(token_id))
            instance_ids.append(instance_id)
            wm = torch.randint(0, 2, (self.n_bits,))
            watermarks.append(wm)
            wmids.append(sum([2**i * wm[i] for i in range(self.n_bits)]))
            labels.append(label)

        padded = pad_sequence(token_ids, batch_first=True, padding_value=self.PAD_ID)
        mask = padded == self.PAD_ID

        return (
            padded,
            torch.tensor(lengths, dtype=torch.long),
            mask,
            instance_ids,
            torch.stack(watermarks, dim=0),
            torch.tensor(wmids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


class DynamicWMCollatorMLM:
    PAD_ID = 0
    MASK_ID = 1

    def __init__(
        self,
        n_bits: int,
        vocab_size: int,
        nomask_token_ids: Optional[Set[int]] = None,
        mask_ratio: float = 0.15,
        replace_prob: float = 0.1,
        retain_prob: float = 0.1,
    ) -> None:
        self.n_bits = n_bits
        self.vocab_size = vocab_size
        self.mask_ratio = mask_ratio
        self.replace_prob = replace_prob
        self.retain_prob = retain_prob

        if nomask_token_ids is None:
            self.nomask_ids = self._default_nomask_token_ids()
        else:
            self.nomask_ids = nomask_token_ids

    def _default_nomask_token_ids(self) -> Set[int]:
        return {0, 1, 2, 3}

    def _mask_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        x, y = token_ids.clone(), token_ids.clone()
        full_mask = torch.rand(x.shape) < self.mask_ratio

        for t in self.nomask_ids:
            full_mask &= x != t

        unchanged_mask = full_mask & (torch.rand(x.shape) < self.retain_prob)
        random_mask = full_mask & (torch.rand(x.shape) < self.replace_prob)
        random_token_idx = torch.nonzero(random_mask, as_tuple=True)
        random_tokens = torch.randint(
            0, self.vocab_size, (len(random_token_idx[0]),), device=x.device
        )
        mask_mask = full_mask & (~unchanged_mask) & (~random_mask)

        x = torch.masked_fill(x, mask_mask, self.MASK_ID)
        x[random_token_idx] = random_tokens
        y = torch.masked_fill(y, ~full_mask, self.PAD_ID)

        return x, y

    def __call__(self, batch):
        instance_ids = []
        token_ids = []
        lengths = []
        watermarks = []
        wmids = []
        for instance_id, token_id in batch:
            token_ids.append(torch.tensor(token_id, dtype=torch.long))
            lengths.append(len(token_id))
            instance_ids.append(instance_id)
            wm = torch.randint(0, 2, (self.n_bits,))
            watermarks.append(wm)
            wmids.append(sum([2**i * wm[i] for i in range(self.n_bits)]))

        padded = pad_sequence(token_ids, batch_first=True, padding_value=self.PAD_ID)
        mask = padded == self.PAD_ID

        # MLM data generation
        mlm_x, mlm_y = self._mask_tokens(padded)

        return (
            padded,
            torch.tensor(lengths, dtype=torch.long),
            mask,
            instance_ids,
            torch.stack(watermarks, dim=0),
            torch.tensor(wmids, dtype=torch.long),
            mlm_x,
            mlm_y,
        )


class WMDetectionCollator:
    def __init__(self, pad_idx: int = 0) -> None:
        self.pad_idx = pad_idx

    def __call__(self, batch):
        token_ids = []
        lengths = []
        labels = []

        for token_id, label in batch:
            token_ids.append(torch.tensor(token_id, dtype=torch.long))
            lengths.append(len(token_id))
            labels.append(label)

        padded = pad_sequence(token_ids, batch_first=True, padding_value=self.pad_idx)
        mask = padded == self.pad_idx

        return (
            padded,
            torch.tensor(lengths, dtype=torch.long),
            mask,
            torch.tensor(labels, dtype=torch.long),
        )


class DewatermarkingCollator:
    def __init__(self, pad_idx: int = 0, batch_first: bool = False) -> None:
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def __call__(self, batch):
        src_token_ids = []
        tgt_token_ids = []

        for src, tgt in batch:
            src_token_ids.append(torch.tensor(src, dtype=torch.long))
            tgt_token_ids.append(torch.tensor(tgt, dtype=torch.long))

        src_token_ids = pad_sequence(
            src_token_ids, padding_value=self.pad_idx, batch_first=self.batch_first
        )
        tgt_token_ids = pad_sequence(
            tgt_token_ids, padding_value=self.pad_idx, batch_first=self.batch_first
        )

        return src_token_ids, tgt_token_ids
