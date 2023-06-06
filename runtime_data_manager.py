import json
import random
from transformers import RobertaTokenizer
from copy import deepcopy
from typing import List, Tuple, Dict, Optional

from code_tokenizer import CodeTokenizer
from code_transform_provider import CodeTransformProvider
from data_processing import (DataInstance, TokenIdCollator, TokenIdCollatorCodeBert,
                             CodeVocab)
from ropgen_transform.py.var_name_style import normalize_name


class InMemoryJitRuntimeDataManager:
    def __init__(self,
                 transform_computer: CodeTransformProvider,
                 original_instances: List[DataInstance],
                 lang: str = 'c'):
        self.original_instances = {
            instance.id: instance
            for instance in original_instances
        }
        self.lang = lang
        self.vocab: CodeVocab = None
        self.transform_mask: Dict = None
        self.all_varnames: List[str] = None
        self.varnames_per_file: Dict = None

        self.code_tokenizer = CodeTokenizer(lang=lang)
        self.collator = TokenIdCollator()

        self.transform_computer = transform_computer
        self.all_transform_keys = self.transform_computer.get_transform_keys()
        self.transform_key_to_idx = {k: i for i, k in enumerate(self.all_transform_keys)}

        self.transformed_instances = dict()

        self.tokenizer: RobertaTokenizer = None

    def get_transform_keys(self):
        return self.all_transform_keys

    def get_transform_capacity(self):
        return len(self.all_transform_keys)

    def register_vocab(self, vocab: CodeVocab):
        self.vocab = vocab

    def register_tokenizer(self, tokenizer: RobertaTokenizer):
        self.tokenizer = tokenizer
        self.collator = TokenIdCollatorCodeBert(tokenizer.pad_token_id)

    def load_varname_dict(self, json_fpath: str):
        res_dict = json.load(open(json_fpath, 'r'))
        self.all_varnames = res_dict['all_variable_names']
        self.varnames_per_file = res_dict['variable_names_per_file']

    def load_transform_mask(self, json_fpath: str):
        loaded = json.load(open(json_fpath, 'r'))
        for instance_key in loaded.keys():
            # convert list-of-list to list-of-tuple
            loaded[instance_key] = [tuple(x) for x in loaded[instance_key]]
        self.transform_mask = loaded

    def get_original_instance(self, instance_id: str) -> DataInstance:
        return self.original_instances[instance_id]

    def get_original_instances(self, instance_ids: List[str]) -> List[DataInstance]:
        return [self.get_original_instance(instance_id) for instance_id in instance_ids]

    def _jit_transform(self, instance_id: str, key_id: int):
        keys = self.all_transform_keys[key_id]
        original_instance = self.get_original_instance(instance_id)
        assert original_instance.id == instance_id

        code = original_instance.source
        transformed_code = self.transform_computer.code_transform(code, keys)
        code_tokens, word_tokens = self.code_tokenizer.get_tokens(transformed_code)
        return DataInstance(id=instance_id,
                            source=transformed_code,
                            source_tokens=code_tokens,
                            tokens=word_tokens,
                            transform_keys=keys)

    def register_transformed_codes(self, instances: List[DataInstance]):
        for instance in instances:
            if len(instance.tokens) == 0:
                raise ValueError(f'empty instance: {repr(instance)}')
            instance_id = instance.id
            transform_keys = instance.transform_keys
            assert transform_keys is not None
            transform_id = self.transform_key_to_idx[tuple(transform_keys)]

            if instance_id not in self.transformed_instances:
                self.transformed_instances[instance_id] = dict()
            instance_dict = self.transformed_instances[instance_id]

            assert transform_id not in instance_dict
            instance_dict[transform_id] = instance

    def _get_cached_transformed_instances(self, instance_id: str,
                                          selected_transform: int) -> DataInstance:
        instances = self.transformed_instances[instance_id]
        return instances[selected_transform]

    def get_transformed_codes_by_pred(
            self, instance_ids: List[str],
            selected_transforms: List[int]) -> List[DataInstance]:

        # jit code transform
        instances_to_jit = []
        for iid, tid in zip(instance_ids, selected_transforms):
            if (iid not in self.transformed_instances
                    or tid not in self.transformed_instances[iid]):
                instances_to_jit.append((iid, tid))

        transformed_instances = []
        for iid, tid in instances_to_jit:
            transformed_instances.append(self._jit_transform(iid, tid))
        self.register_transformed_codes(transformed_instances)

        # collect results
        instances = []
        for iid, tid in zip(instance_ids, selected_transforms):
            instances.append(self._get_cached_transformed_instances(iid, tid))

        return instances

    def _jit_varname_substitution(self,
                                  instance: DataInstance,
                                  new_token: str,
                                  old_token: Optional[str] = None):
        new_instance = deepcopy(instance)
        varnames = self.varnames_per_file[instance.id]

        if len(varnames) == 0:
            return new_instance, ('no feasible substitution', '')

        src_var = random.choice(varnames) if old_token is None else old_token

        new_code = self.transform_computer.variable_substitution(
            new_instance.source, src_var, new_token)

        source_tokens, tokens = self.code_tokenizer.get_tokens(new_code)
        new_instance = DataInstance(id=new_instance.id,
                                    source=new_code,
                                    source_tokens=source_tokens,
                                    tokens=tokens,
                                    transform_keys=new_instance.transform_keys)

        return new_instance, (src_var, new_token)

    def varname_transform_on_instances_bert(
            self, instances: List[DataInstance],
            word_preds: List[int]) -> Tuple[List[DataInstance], List[Tuple[str, str]]]:
        assert self.tokenizer is not None

        new_instances = []
        updates = []
        for instance, word_pred in zip(instances, word_preds):
            new_word = self.tokenizer._convert_id_to_token(word_pred).replace('Ġ', '')
            new_instance, update = self._jit_varname_substitution(instance, new_word)
            new_instances.append(new_instance)
            updates.append(update)

        return new_instances, updates

    def varname_transform_on_instances(
            self, instances: List[DataInstance],
            word_preds: List[int]) -> Tuple[List[DataInstance], List[Tuple[str, str]]]:
        assert self.vocab is not None

        new_instances = []
        updates = []
        for instance, word_pred in zip(instances, word_preds):
            new_word = self.vocab.get_token_by_id(word_pred)
            new_instance, update = self._jit_varname_substitution(instance, new_word)
            new_instances.append(new_instance)
            updates.append(update)

        return new_instances, updates

    def load_to_tensor_bert(self, instances: List[DataInstance]):
        batch = []
        for instance in instances:
            tokens = self.tokenizer.tokenize(' '.join(instance.tokens))[:510]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            token_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids)
            batch.append(token_ids)

        return self.collator(batch)

    def load_to_tensor(self, instances: List[DataInstance]):
        batch = [
            self.vocab.convert_tokens_to_ids(instance.tokens)[:512]
            for instance in instances
        ]

        return self.collator(batch)

    def _get_feasible_transform_ids_for_file(self, instance_id: str):
        feasibles = self.transform_mask[instance_id]
        return [self.transform_key_to_idx[tuple(x)] for x in feasibles]

    def get_feasible_transform_ids(self, instance_ids: List[str]):
        masks = []
        for instance_id in instance_ids:
            masks.append(self._get_feasible_transform_ids_for_file(instance_id))

        return masks

    def _conditional_jit_varname_substitution(self, instance: DataInstance,
                                              new_token: str, update: Tuple[str, str]):
        new_instance = deepcopy(instance)
        varnames = self.varnames_per_file[instance.id]

        if len(varnames) == 0:
            return new_instance, ('no feasible substitution', '')

        src_var = random.choice(varnames)
        src_var_str = normalize_name(src_var).lower()

        prev_src_str = normalize_name(update[0]).lower()
        prev_dst_str = normalize_name(update[1]).lower()

        # chosen var has been modified in the previous transform
        if src_var_str == prev_src_str:
            actual_src = prev_dst_str
        else:
            actual_src = src_var_str

        new_code = self.transform_computer.variable_substitution(
            new_instance.source, actual_src, new_token)

        source_tokens, tokens = self.code_tokenizer.get_tokens(new_code)
        new_instance = DataInstance(id=new_instance.id,
                                    source=new_code,
                                    source_tokens=source_tokens,
                                    tokens=tokens,
                                    transform_keys=new_instance.transform_keys)

        return new_instance, (src_var, new_token)

    def rewatermark_varname_transform(
        self, instances: List[DataInstance], word_preds: List[int],
        prev_updates: List[Tuple[str]]
    ) -> Tuple[List[DataInstance], List[Tuple[str, str]]]:
        assert self.vocab is not None

        new_instances = []
        updates = []
        for instance, word_pred, prev_update in zip(instances, word_preds, prev_updates):
            new_word = self.vocab.get_token_by_id(word_pred)
            new_instance, update = self._conditional_jit_varname_substitution(
                instance, new_word, prev_update)
            new_instances.append(new_instance)
            updates.append(update)

        return new_instances, updates
