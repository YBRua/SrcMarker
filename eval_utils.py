import json
import copy
import torch
import random

from data_processing import DataInstance
from code_tokenizer import CodeTokenizer
from varname_utils import normalize_name
from code_transform_provider import CodeTransformProvider

from typing import Dict, List, Optional, Tuple


def compute_msg_acc(inputs: torch.Tensor,
                    target: torch.Tensor,
                    n_bits: int,
                    reduce: bool = False):
    # inputs: B, n_bits
    per_sample_hits = torch.sum((inputs == target), dim=-1)  # B
    msg_acc = torch.sum(per_sample_hits == n_bits).item()
    if reduce:
        msg_acc /= inputs.size(0)
    return msg_acc


class JitAdversarialTransformProvider:

    def __init__(self,
                 transform_computer: CodeTransformProvider,
                 transforms_per_file: str = './datasets/transforms_per_file.json',
                 varname_path: str = './datasets/variable_names.json',
                 lang: str = 'cpp') -> None:
        self.t_combos = json.load(open(transforms_per_file, 'r', encoding='utf-8'))
        varname_dict = json.load(open(varname_path, 'r', encoding='utf-8'))
        self.all_varnames = varname_dict['all_variable_names']
        self.varnames_per_file = varname_dict['variable_names_per_file']

        self.all_transform_keys = transform_computer.get_transform_keys()
        self.transform_key_to_idx = {k: i for i, k in enumerate(self.all_transform_keys)}

        self.pipeline = transform_computer.pipeline
        self.transform_computer = transform_computer
        self.tokenizer = CodeTokenizer(lang)

    def _default_transform_keys(self, full_file_dict: Dict):
        default_keys = []
        for t_name in self.pipeline.get_transformer_names():
            t = self.pipeline.get_transformer(t_name)
            theoreticals = t.get_available_transforms()
            if t_name not in full_file_dict:
                # current transform type is not applicable on the instance
                # use default value (first one)
                default_keys.append(theoreticals[0])
            elif len(full_file_dict[t_name]) == len(theoreticals):
                # all transform types are applicable and represent a different instance
                # use the first one, but note that this is lossy
                default_keys.append(theoreticals[0])
            else:
                for th in theoreticals:
                    if th not in full_file_dict[t_name]:
                        default_keys.append(th)
                        break
        return default_keys

    def adv_style_transform(self, instance: DataInstance, n_transforms: int) -> int:
        i_dict = self.t_combos[instance.id]
        instance_dict = copy.deepcopy(i_dict)

        transform_names = self.pipeline.get_transformer_names()

        if instance.transform_keys is None:
            current_keys = self._default_transform_keys(instance_dict)
        else:
            current_keys = instance.transform_keys

        for t_name in transform_names:
            t = self.pipeline.get_transformer(t_name)
            theoreticals = t.get_available_transforms()
            if t_name not in instance_dict:
                # current transform type is not applicable on the instance
                # use default value (first one)
                instance_dict[t_name] = [theoreticals[0]]
            else:
                # current transform type is applicable,
                # each value in full_file_dict represents in a different instance
                # so we still need to add a default value that stands for "no changes"
                for th in theoreticals:
                    if th not in instance_dict[t.name]:
                        instance_dict[t_name].append(th)
                        break

        # select n_transforms transforms from full_file_dict
        selected_keys = [0] * len(transform_names)
        budget = n_transforms

        # shuffle transform types so that a random type is selected
        shuffled = list(range(len(transform_names)))
        random.shuffle(shuffled)

        for i in range(len(transform_names)):
            t_name = transform_names[shuffled[i]]
            if t_name in instance_dict:
                if budget > 0 and len(instance_dict[t_name]) > 1:
                    # can selected a (different) transform
                    keys = instance_dict[t_name]
                    copied_keys = copy.deepcopy(keys)
                    copied_keys.remove(current_keys[shuffled[i]])
                    selected_keys[shuffled[i]] = random.choice(copied_keys)
                    budget -= 1
                else:
                    # no budget, or no other different applicable transform
                    # retain the original transform
                    selected_keys[shuffled[i]] = current_keys[shuffled[i]]
            else:
                selected_keys[shuffled[i]] = current_keys[shuffled[i]]
        if budget > 0:
            print('Warning: budget not used up')
        return self.transform_key_to_idx[tuple(selected_keys)]

    def get_adv_style_transforms(self, instances: List[DataInstance],
                                 n_transforms: int) -> List[int]:
        selected = []
        for instance in instances:
            selected.append(self.adv_style_transform(instance, n_transforms))

        return selected

    def _jit_varname_substitution(self,
                                  instance: DataInstance,
                                  new_token: str,
                                  old_token: Optional[str] = None):
        new_instance = copy.deepcopy(instance)
        varnames = self.varnames_per_file[instance.id]

        if len(varnames) == 0:
            return new_instance, ('no feasible substitution', '')

        src_var = random.choice(varnames) if old_token is None else old_token

        new_code = self.transform_computer.variable_substitution(
            new_instance.source, src_var, new_token)

        source_tokens, tokens = self.tokenizer.get_tokens(new_code)
        new_instance = DataInstance(id=new_instance.id,
                                    source=new_code,
                                    source_tokens=source_tokens,
                                    tokens=tokens,
                                    transform_keys=new_instance.transform_keys)

        return new_instance, (src_var, new_token)

    def redo_varname_transform(self, instances: List[DataInstance],
                               updates: List[Tuple[str, str]]) -> List[DataInstance]:
        new_instances = []
        for instance, update in zip(instances, updates):
            new_instance, _ = self._jit_varname_substitution(instance, update[1],
                                                             update[0])
            new_instances.append(new_instance)
        return new_instances

    def adv_varname_transform(
            self,
            instances: List[DataInstance],
            budget: Optional[int] = None,
            proportion: Optional[float] = None,
            var_updates: Optional[List[Tuple[str]]] = None) -> List[DataInstance]:
        if budget is None and proportion is None:
            raise ValueError('either budget or proportion must be specified')
        new_instances = []
        all_adv_updates = []
        for i, instance in enumerate(instances):
            file_varnames = self.varnames_per_file[instance.id]
            if len(file_varnames) == 0:
                # no replacable variable names
                new_instances.append(instance)
                all_adv_updates.append([])
                continue

            # non-targeted variable substitution
            if budget is None:
                # guaranteed to have at least one variable name changed
                budget = max(int(len(file_varnames) * proportion), 1)
            src_vars = random.sample(list(file_varnames), budget)
            adv_updates = []
            for src_var in src_vars:
                if var_updates is not None:
                    update = var_updates[i]
                    # if src_var has been changed when embedding watermark
                    # update src to the new name
                    if (normalize_name(src_var).lower() == normalize_name(
                            update[0]).lower()):
                        src_var = update[1]

                dst_var = random.choice(self.all_varnames)
                instance, u = self._jit_varname_substitution(instance, dst_var, src_var)
                adv_updates.append(u)
            all_adv_updates.append(adv_updates)
            new_instances.append(instance)
        return new_instances, all_adv_updates
