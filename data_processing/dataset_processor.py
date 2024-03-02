import os
import json

from tqdm import tqdm
from multiprocessing import Pool

from .code_vocab import CodeVocab
from .data_instance import (
    DataInstance,
    WMDetectionDataInstance,
    DewatermarkingDataInstance,
)
from .code_dataset import (
    JsonlCodeWatermarkDataset,
    JsonlWMDetectionDataset,
    JsonlTaskedCodeWatermarkDataset,
    JsonlDewatermarkingDataset,
)
from benchmark_mbxp import compose_function_java, compose_function_javascript
from code_tokenizer import CodeTokenizer
from typing import List, Dict


class JsonlDatasetProcessor:
    def __init__(self, lang: str = "cpp") -> None:
        self.lang = lang
        # self.code_tokenizer = CodeTokenizer(lang=lang)

    def load_jsonl(self, data_dir: str, split: str, show_progress: bool = True) -> List:
        fpath = os.path.join(data_dir, f"{split}.jsonl")
        with open(fpath, "r", encoding="utf-8") as fi:
            lines = fi.readlines()
        objs = [json.loads(line) for line in lines]

        instances = []
        args = []
        for i, obj in enumerate(objs):
            args.append((i, obj, split))

        with Pool(os.cpu_count() // 2) as pool:
            result = pool.imap(self._mp_process_instances_wrapper, args)

            if show_progress:
                result = tqdm(result, desc=f"{split:10}", total=len(objs))

            for res in result:
                instances.append(res)

        return instances

    def load_raw_jsonl(self, data_dir: str, split: str):
        fpath = os.path.join(data_dir, f"{split}.jsonl")
        with open(fpath, "r", encoding="utf-8") as fi:
            lines = fi.readlines()
        objs = [json.loads(line) for line in lines]
        return objs

    def _mp_process_instances_wrapper(self, args):
        return self._process_instance(*args)

    def load_jsonls(self, data_dir: str, show_progress: bool = True) -> Dict[str, List]:
        train_instances = self.load_jsonl(
            data_dir, split="train", show_progress=show_progress
        )
        valid_instances = self.load_jsonl(
            data_dir, split="valid", show_progress=show_progress
        )
        test_instances = self.load_jsonl(
            data_dir, split="test", show_progress=show_progress
        )

        return {
            "train": train_instances,
            "valid": valid_instances,
            "test": test_instances,
        }

    def _process_instance(self, id: int, data_obj: Dict, split: str):
        raise NotImplementedError()

    def build_dataset(self, instances: List, vocab: CodeVocab):
        raise NotImplementedError()

    def build_vocab(self, instances: List) -> CodeVocab:
        raise NotImplementedError()


class JsonlWMDatasetProcessor(JsonlDatasetProcessor):
    def __init__(self, lang: str = "cpp") -> None:
        super().__init__(lang)

    def _process_instance(self, id: int, data_obj: Dict, split: str) -> DataInstance:
        source = data_obj["original_string"]
        code_tokenizer = CodeTokenizer(lang=self.lang)
        code_tokens, word_tokens = code_tokenizer.get_tokens(source)
        return DataInstance(
            id=f"{split}#{id}",
            source=source,
            source_tokens=code_tokens,
            tokens=word_tokens,
            transform_keys=None,
        )

    def build_dataset(
        self, instances: List[DataInstance], vocab: CodeVocab
    ) -> JsonlCodeWatermarkDataset:
        return JsonlCodeWatermarkDataset(instances, vocab)

    def build_vocab(self, instances: List[DataInstance]) -> CodeVocab:
        vocab = CodeVocab()
        for instance in instances:
            for tok in instance.tokens:
                vocab.add_word(tok)

        return vocab

    def load_jsonls(
        self, data_dir: str, show_progress: bool = True
    ) -> Dict[str, List[DataInstance]]:
        return super().load_jsonls(data_dir, show_progress)

    def _process_instance_fast(
        self, id: int, data_obj: Dict, split: str
    ) -> DataInstance:
        source = data_obj["original_string"]
        return DataInstance(
            id=f"{split}#{id}",
            source=source,
            source_tokens=None,
            tokens=None,
            transform_keys=None,
        )

    def _load_jsonl_fast(self, data_dir: str, split: str, show_progress: bool = True):
        fpath = os.path.join(data_dir, f"{split}.jsonl")
        with open(fpath, "r", encoding="utf-8") as fi:
            lines = fi.readlines()
        objs = [json.loads(line) for line in lines]

        instances = []
        if show_progress:
            objs = tqdm(objs, desc=f"{split:10}")

        for i, obj in enumerate(objs):
            instances.append(self._process_instance_fast(i, obj, split))

        return instances

    def load_jsonls_fast(
        self, data_dir: str, show_progress: bool = True
    ) -> Dict[str, List[DataInstance]]:
        # only loads source code strings, no tokenization
        # for preprocessing only
        train_instances = self._load_jsonl_fast(
            data_dir, split="train", show_progress=show_progress
        )
        valid_instances = self._load_jsonl_fast(
            data_dir, split="valid", show_progress=show_progress
        )
        test_instances = self._load_jsonl_fast(
            data_dir, split="test", show_progress=show_progress
        )

        return {
            "train": train_instances,
            "valid": valid_instances,
            "test": test_instances,
        }


class JsonlTaskedWMDatasetProcessor(JsonlDatasetProcessor):
    def __init__(self, label_key: str, lang: str = "cpp") -> None:
        super().__init__(lang)
        self.label_key = label_key

    def _process_instance(self, id: int, data_obj: Dict, split: str) -> DataInstance:
        source = data_obj["original_string"]
        label = data_obj[self.label_key]
        code_tokenizer = CodeTokenizer(lang=self.lang)
        code_tokens, word_tokens = code_tokenizer.get_tokens(source)
        return DataInstance(
            id=f"{split}#{id}",
            source=source,
            source_tokens=code_tokens,
            tokens=word_tokens,
            task_label=label,
            transform_keys=None,
        )

    def build_dataset(
        self,
        instances: List[DataInstance],
        vocab: CodeVocab,
        label2idx: Dict[str, int],
        is_validation: bool = False,
    ) -> JsonlTaskedCodeWatermarkDataset:
        return JsonlTaskedCodeWatermarkDataset(
            instances, vocab, label2idx, is_validation
        )

    def build_label_dict(self, instances: List[DataInstance]):
        label2idx = dict()
        idx2label = list()
        for instance in instances:
            if instance.task_label not in label2idx:
                label2idx[instance.task_label] = len(idx2label)
                idx2label.append(instance.task_label)
        return label2idx, idx2label

    def build_vocab(self, instances: List[DataInstance]) -> CodeVocab:
        vocab = CodeVocab()
        for instance in instances:
            for tok in instance.tokens:
                vocab.add_word(tok)

        return vocab

    def load_jsonls(
        self, data_dir: str, show_progress: bool = True
    ) -> Dict[str, List[DataInstance]]:
        return super().load_jsonls(data_dir, show_progress)


class JsonlDetectionDatasetProcessor(JsonlDatasetProcessor):
    def __init__(self, lang: str = "cpp") -> None:
        super().__init__(lang)

    def load_jsonls(
        self, data_dir: str, show_progress: bool = True
    ) -> Dict[str, List[WMDetectionDataInstance]]:
        return super().load_jsonls(data_dir, show_progress)

    def _process_instance(
        self, id: int, obj: Dict, split: str
    ) -> WMDetectionDataInstance:
        source = obj["after_watermark"]
        contains_watermark = obj["contains_watermark"]
        code_tokenizer = CodeTokenizer(lang=self.lang)
        _, word_tokens = code_tokenizer.get_tokens(source)
        return WMDetectionDataInstance(
            id=f"{split}#{id}",
            source=source,
            tokens=word_tokens,
            label=contains_watermark,
        )

    def build_dataset(
        self, instances: List, vocab: CodeVocab
    ) -> JsonlWMDetectionDataset:
        return JsonlWMDetectionDataset(instances, vocab)

    def build_vocab(self, instances: List[WMDetectionDataInstance]) -> CodeVocab:
        vocab = CodeVocab()
        for instance in instances:
            for tok in instance.tokens:
                vocab.add_word(tok)

        return vocab

    def _process_instance_fast(
        self, id: int, data_obj: Dict, split: str
    ) -> WMDetectionDataInstance:
        source = data_obj["after_watermark"]
        contains_watermark = data_obj["contains_watermark"]
        tokens = source.strip().split(" ")
        return WMDetectionDataInstance(
            id=f"{split}#{id}", source=source, tokens=tokens, label=contains_watermark
        )

    def _load_jsonl_fast(self, data_dir: str, split: str, show_progress: bool = True):
        fpath = os.path.join(data_dir, f"{split}.jsonl")
        with open(fpath, "r", encoding="utf-8") as fi:
            lines = fi.readlines()
        objs = [json.loads(line) for line in lines]

        instances = []
        if show_progress:
            objs = tqdm(objs, desc=f"{split:10}")

        for i, obj in enumerate(objs):
            instances.append(self._process_instance_fast(i, obj, split))

        return instances

    def load_jsonls_fast(
        self, data_dir: str, show_progress: bool = True
    ) -> Dict[str, List[WMDetectionDataInstance]]:
        # only loads source code strings, no tokenization
        # for awt outputs only
        train_instances = self._load_jsonl_fast(
            data_dir, split="train", show_progress=show_progress
        )
        valid_instances = self._load_jsonl_fast(
            data_dir, split="valid", show_progress=show_progress
        )
        test_instances = self._load_jsonl_fast(
            data_dir, split="test", show_progress=show_progress
        )

        return {
            "train": train_instances,
            "valid": valid_instances,
            "test": test_instances,
        }


class JsonlDewatermarkingDatasetProcessor(JsonlDatasetProcessor):
    def __init__(self, lang: str = "cpp") -> None:
        super().__init__(lang)

    def load_jsonls(
        self, data_dir: str, show_progress: bool = True
    ) -> Dict[str, List[DewatermarkingDataInstance]]:
        return super().load_jsonls(data_dir, show_progress)

    def _process_instance(self, id: int, obj: Dict, split: str):
        source = obj["after_watermark"]
        target = obj["original_string"]
        code_tokenizer = CodeTokenizer(lang=self.lang)
        _, source_tokens = code_tokenizer.get_tokens(source)
        _, target_tokens = code_tokenizer.get_tokens(target)
        return DewatermarkingDataInstance(
            id=f"{split}#{id}",
            source=source,
            source_tokens=source_tokens,
            target=target,
            target_tokens=target_tokens,
        )

    def build_dataset(
        self, instances: List, vocab: CodeVocab
    ) -> JsonlDewatermarkingDataset:
        return JsonlDewatermarkingDataset(instances, vocab)

    def build_vocab(self, instances: List[DewatermarkingDataInstance]) -> CodeVocab:
        vocab = CodeVocab()
        for instance in instances:
            for tok in instance.source_tokens:
                vocab.add_word(tok)
            for tok in instance.target_tokens:
                vocab.add_word(tok)

        return vocab
