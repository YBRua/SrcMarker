import torch
from torch.nn.utils.rnn import pad_sequence


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
            wm = torch.randint(0, 2, (self.n_bits, ))
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
            wm = torch.randint(0, 2, (self.n_bits, ))
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
