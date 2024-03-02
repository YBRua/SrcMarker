from torch.utils.data import Dataset

from .code_vocab import CodeVocab
from .data_instance import (
    DataInstance,
    WMDetectionDataInstance,
    DewatermarkingDataInstance,
)
from transformers import RobertaTokenizer
from typing import List, Dict

MAX_TOKEN_LEN = 512


class JsonlWMDetectionDataset(Dataset):
    def __init__(self, instances: List[WMDetectionDataInstance], vocab: CodeVocab):
        super().__init__()
        self.instances = instances
        self.vocab = vocab

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index: int):
        instance = self.instances[index]
        return (
            self.vocab.convert_tokens_to_ids(instance.tokens)[:MAX_TOKEN_LEN],
            1 if instance.label else 0,
        )


class JsonlDewatermarkingDataset(Dataset):
    def __init__(self, instances: List[DewatermarkingDataInstance], vocab: CodeVocab):
        super().__init__()
        self.instances = instances
        self.vocab = vocab

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index: int):
        instance = self.instances[index]
        src = self.vocab.convert_tokens_to_ids(instance.source_tokens)[: 400 - 2]
        src = [self.vocab.bos_idx()] + src + [self.vocab.eos_idx()]
        tgt = self.vocab.convert_tokens_to_ids(instance.target_tokens)[: 400 - 2]
        tgt = [self.vocab.bos_idx()] + tgt + [self.vocab.eos_idx()]

        return src, tgt


class JsonlTaskedCodeWatermarkDataset(Dataset):
    def __init__(
        self,
        instances: List[DataInstance],
        vocab: CodeVocab,
        label_dict: Dict[str, int],
        is_validation: bool = False,
    ):
        super().__init__()
        self.instances = instances
        self.vocab = vocab
        self.label_dict = label_dict
        self.is_validation = is_validation

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index: int):
        instance = self.instances[index]
        if self.is_validation:
            label = 0
        else:
            label = self.label_dict[instance.task_label]
        return (
            instance.id,
            self.vocab.convert_tokens_to_ids(instance.tokens)[:MAX_TOKEN_LEN],
            label,
        )


class JsonlCodeWatermarkDataset(Dataset):
    def __init__(self, instances: List[DataInstance], vocab: CodeVocab):
        super().__init__()
        self.instances = instances
        self.vocab = vocab

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index: int):
        instance = self.instances[index]
        return (
            instance.id,
            self.vocab.convert_tokens_to_ids(instance.tokens)[:MAX_TOKEN_LEN],
        )


class BertCodeWatermarkDataset(Dataset):
    def __init__(
        self, instances: List[DataInstance], tokenizer: RobertaTokenizer
    ) -> None:
        super().__init__()
        self.instances = instances
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index: int):
        instance = self.instances[index]
        tokens = self.tokenizer.tokenize(" ".join(instance.tokens))[:510]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids)
        return (instance.id, token_ids)
