import os
import sys
import json
import tree_sitter
from tqdm import tqdm
from collections import defaultdict
from itertools import product
from multiprocessing import Pool

from data_processing import JsonlDatasetProcessor
from code_transform_provider import CodeTransformProvider
import mutable_tree.transformers as ast_transformers

from typing import List


def collect_tokens(root: tree_sitter.Node) -> List[str]:
    tokens: List[str] = []

    def _collect_tokens(node: tree_sitter.Node):
        if node.child_count == 0:
            tokens.append(node.text.decode())

        for ch in node.children:
            _collect_tokens(ch)

    _collect_tokens(root)
    return tokens


def mp_transform_worker_impl(instance_id: str, code: str, key: str):
    feasible = False
    transform_name = key.split('.')[0]
    new_code = transform_computer.code_transform(code, [key])

    if LANG == 'java':
        code_wrapped = f'public class Test {{\n{code}\n}}'
        new_code_wrapped = f'public class Test {{\n{new_code}\n}}'
    else:
        code_wrapped = code
        new_code_wrapped = new_code

    code_tree = parser.parse(bytes(code_wrapped, 'utf-8'))
    new_code_tree = parser.parse(bytes(new_code_wrapped, 'utf-8'))

    # check if the code is still valid
    try:
        transform_computer.to_mutable_tree(new_code)
    except Exception:
        print(f'failed to parse {instance_id} {transform_name} {key}')
        print(new_code)
        print()
        print(code)
        print()
        feasible = False

    code_tokens = collect_tokens(code_tree.root_node)
    new_code_tokens = collect_tokens(new_code_tree.root_node)

    if len(code_tokens) != len(new_code_tokens):
        feasible = True

    for i in range(len(code_tokens)):
        if code_tokens[i] != new_code_tokens[i]:
            feasible = True
            break

    return (instance_id, transform_name, key, feasible)


def mp_transform_worker(job_args):
    try:
        return mp_transform_worker_impl(*job_args)
    except Exception as e:
        # for arg in job_args:
        #     print(arg)
        # import traceback
        # traceback.print_exc()
        print(type(e).__name__, e)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 1:
        print('usage: python collect_feasible_transforms_jsonl.py <dataset>')

    DATASET = args[0]
    if DATASET in {'csn_java', 'github_java_funcs'}:
        LANG = 'java'
    elif DATASET in {'github_c_funcs'}:
        LANG = 'cpp'
    else:
        raise ValueError(f'Unknown dataset: {DATASET}')

    N_BITS = 4
    DATA_DIR = f'./datasets/{DATASET}'
    do_scan = True  # skip scanning if False
    do_compute_mask = True  # skip computing mask if False

    code_transformers: List[ast_transformers.CodeTransformer] = [
        ast_transformers.IfBlockSwapTransformer(),
        ast_transformers.CompoundIfTransformer(),
        ast_transformers.ConditionTransformer(),
        ast_transformers.LoopTransformer(),
        ast_transformers.UpdateTransformer(),
        ast_transformers.SameTypeDeclarationTransformer(),
        ast_transformers.VarDeclLocationTransformer(),
        ast_transformers.VarInitTransformer(),
        ast_transformers.VarNameStyleTransformer()
    ]

    parser = tree_sitter.Parser()
    parser_lang = tree_sitter.Language('./parser/languages.so', LANG)
    parser.set_language(parser_lang)
    transform_computer = CodeTransformProvider(LANG, parser, code_transformers)

    data_processor = JsonlDatasetProcessor(LANG)
    instances = data_processor.load_jsonls_fast(DATA_DIR, show_progress=False)
    train_instances = instances['train']
    valid_instances = instances['valid']
    test_instances = instances['test']

    all_instances = train_instances + valid_instances + test_instances

    transforms_per_file = defaultdict(dict)
    if do_scan:
        jobs = []
        for instance in all_instances:
            for t in code_transformers:
                keys = t.get_available_transforms()
                for key in keys:
                    jobs.append((instance.id, instance.source, key))

        with Pool(os.cpu_count() // 2) as p:
            results = p.imap(mp_transform_worker, jobs)
            for res in tqdm(results, total=len(jobs)):
                if res is None:
                    raise RuntimeError('transform worker failed')

                (iid, t_name, key, feasible) = res
                instance_dict = transforms_per_file[iid]
                if t_name not in instance_dict:
                    instance_dict[t_name] = []
                transform_list = instance_dict[t_name]
                if feasible:
                    transform_list.append(key)
        json.dump(transforms_per_file,
                  open(f'./datasets/transforms_per_file_{DATASET}.json', 'w'),
                  indent=2)

    if do_compute_mask:
        transforms_per_file = json.load(
            open(f'./datasets/transforms_per_file_{DATASET}.json', 'r'))
        all_keys = []
        for t in code_transformers:
            keys = t.get_available_transforms()
            all_keys.append(keys)

        feasible_transforms_per_file = dict()
        for iid, instance_dict in transforms_per_file.items():
            all_possible_transforms = list(product(*all_keys))

            # find all infeasible transforms
            not_available = []
            for t in code_transformers:
                theoreticals = t.get_available_transforms()

                if t.name not in instance_dict:
                    # all transforms under this category is infeasible
                    feasible_set = set()
                else:
                    feasible_set = set(instance_dict[t.name])

                if len(feasible_set) < len(theoreticals):
                    # at least one transform is infeasible
                    for tt in theoreticals:
                        if tt not in feasible_set:
                            # add first transform as a default value
                            feasible_set.add(tt)
                            break

                not_available.extend([t for t in theoreticals if t not in feasible_set])

            # filter file_list with not_available
            feasibles = []
            for t_sequence in all_possible_transforms:
                if not any([t in not_available for t in t_sequence]):
                    feasibles.append(t_sequence)

            feasible_transforms_per_file[iid] = feasibles

        json.dump(feasible_transforms_per_file,
                  open(f'./datasets/feasible_transform_{DATASET}.json', 'w'),
                  indent=2)

    feasible_transforms_per_file = json.load(
        open(f'./datasets/feasible_transform_{DATASET}.json', 'r'))

    tot_files = 0
    ok_files = 0
    tot_transformed_files = 0
    for insatnce_id, file_list in feasible_transforms_per_file.items():
        tot_files += 1
        tot_transformed_files += len(file_list)
        if len(file_list) >= 2**N_BITS:
            ok_files += 1

    print(f'total files: {tot_files}')
    print(f'ok files: {ok_files} ({ok_files / tot_files * 100:.2f}))')
    print(f'total transformed files: {tot_transformed_files}')
    print(f'average: {tot_transformed_files / tot_files:.2f}')
