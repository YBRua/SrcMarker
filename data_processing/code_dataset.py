from torch.utils.data import Dataset

from .code_vocab import CodeVocab
from .data_instance import DataInstance
from transformers import RobertaTokenizer
from typing import List

MAX_TOKEN_LEN = 512


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
    def __init__(self, instances: List[DataInstance],
                 tokenizer: RobertaTokenizer) -> None:
        super().__init__()
        self.instances = instances
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index: int):
        instance = self.instances[index]
        tokens = self.tokenizer.tokenize(' '.join(instance.tokens))[:510]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids)
        return (instance.id, token_ids)
