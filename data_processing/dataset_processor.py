import os
import json

from tqdm import tqdm
from multiprocessing import Pool

from .code_vocab import CodeVocab
from .data_instance import DataInstance
from .code_dataset import JsonlCodeWatermarkDataset
from code_tokenizer import CodeTokenizer
from typing import List, Dict


class JsonlDatasetProcessor:

    def __init__(self, lang: str = 'cpp') -> None:
        self.lang = lang
        self.code_tokenizer = CodeTokenizer(lang=lang)

    def load_jsonl(self, data_dir: str, split: str, show_progress: bool = True):
        fpath = os.path.join(data_dir, f'{split}.jsonl')
        with open(fpath, 'r', encoding='utf-8') as fi:
            lines = fi.readlines()
        objs = [json.loads(line) for line in lines]

        instances = []
        args = []
        for i, obj in enumerate(objs):
            args.append((i, obj, split))

        with Pool(os.cpu_count() // 2) as pool:
            result = pool.imap(self._mp_process_instances_wrapper, args)

            if show_progress:
                result = tqdm(result, desc=f'{split:10}', total=len(objs))

            for res in result:
                instances.append(res)

        return instances

    def load_raw_jsonl(self, data_dir: str, split: str):
        fpath = os.path.join(data_dir, f'{split}.jsonl')
        with open(fpath, 'r', encoding='utf-8') as fi:
            lines = fi.readlines()
        objs = [json.loads(line) for line in lines]
        return objs

    def _load_jsonl_fast(self, data_dir: str, split: str, show_progress: bool = True):
        fpath = os.path.join(data_dir, f'{split}.jsonl')
        with open(fpath, 'r', encoding='utf-8') as fi:
            lines = fi.readlines()
        objs = [json.loads(line) for line in lines]

        instances = []
        if show_progress:
            objs = tqdm(objs, desc=f'{split:10}')

        for i, obj in enumerate(objs):
            instances.append(self._process_instance_fast(i, obj, split))

        return instances

    def _process_instance_fast(self, id: int, data_obj: Dict, split: str) -> DataInstance:
        source = data_obj['original_string']
        return DataInstance(id=f'{split}#{id}',
                            source=source,
                            source_tokens=None,
                            tokens=None,
                            transform_keys=None)

    def _process_instance(self, id: int, data_obj: Dict, split: str) -> DataInstance:
        source = data_obj['original_string']
        code_tokens, word_tokens = self.code_tokenizer.get_tokens(source)
        return DataInstance(id=f'{split}#{id}',
                            source=source,
                            source_tokens=code_tokens,
                            tokens=word_tokens,
                            transform_keys=None)

    def _mp_process_instances_wrapper(self, args):
        return self._process_instance(*args)

    def load_jsonls(self,
                    data_dir: str,
                    show_progress: bool = True) -> Dict[str, List[DataInstance]]:

        train_instances = self.load_jsonl(data_dir,
                                          split='train',
                                          show_progress=show_progress)
        valid_instances = self.load_jsonl(data_dir,
                                          split='valid',
                                          show_progress=show_progress)
        test_instances = self.load_jsonl(data_dir,
                                         split='test',
                                         show_progress=show_progress)

        return {
            'train': train_instances,
            'valid': valid_instances,
            'test': test_instances
        }

    def load_jsonls_fast(self,
                         data_dir: str,
                         show_progress: bool = True) -> Dict[str, List[DataInstance]]:
        # only loads source code strings, no tokenization
        # for preprocessing only
        train_instances = self._load_jsonl_fast(data_dir,
                                                split='train',
                                                show_progress=show_progress)
        valid_instances = self._load_jsonl_fast(data_dir,
                                                split='valid',
                                                show_progress=show_progress)
        test_instances = self._load_jsonl_fast(data_dir,
                                               split='test',
                                               show_progress=show_progress)

        return {
            'train': train_instances,
            'valid': valid_instances,
            'test': test_instances
        }

    def build_dataset(self, instances: List[DataInstance],
                      vocab: CodeVocab) -> JsonlCodeWatermarkDataset:
        return JsonlCodeWatermarkDataset(instances, vocab)

    def build_vocab(self, instances: List[DataInstance]) -> CodeVocab:
        vocab = CodeVocab()
        for instance in instances:
            for tok in instance.tokens:
                vocab.add_word(tok)

        return vocab
