import os
import sys
import json
import tree_sitter
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool

from data_processing import JsonlWMDatasetProcessor
from code_transform_provider import CodeTransformProvider
import mutable_tree.transformers as ast_transformers
from mutable_tree.stringifiers import JavaScriptStringifier

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


def mp_transform_worker_impl(lang: str, instance_id: str, code: str, key: str):
    code_transformers: List[ast_transformers.CodeTransformer] = [
        ast_transformers.IfBlockSwapTransformer(),
        ast_transformers.CompoundIfTransformer(),
        ast_transformers.ConditionTransformer(),
        ast_transformers.LoopTransformer(),
        ast_transformers.InfiniteLoopTransformer(),
        ast_transformers.UpdateTransformer(),
        ast_transformers.SameTypeDeclarationTransformer(),
        ast_transformers.VarDeclLocationTransformer(),
        ast_transformers.VarInitTransformer(),
        ast_transformers.VarNameStyleTransformer()
    ]

    parser = tree_sitter.Parser()
    parser_lang = tree_sitter.Language('./parser/languages.so', lang)
    parser.set_language(parser_lang)
    transform_computer = CodeTransformProvider(lang, parser, code_transformers)

    feasible = False
    transform_name = key.split('.')[0]
    try:
        new_code = transform_computer.code_transform(code, [key])
    except Exception as e:
        print(f'failed to transform {instance_id} {transform_name} {key}')
        print(f'message: {e}')
        new_code = code
        feasible = False

    if lang == 'java':
        code_wrapped = f'public class Test {{\n{code}\n}}'
        new_code_wrapped = f'public class Test {{\n{new_code}\n}}'
    elif lang == 'cpp' or lang == 'javascript':
        code_wrapped = code
        new_code_wrapped = new_code

    if lang == 'javascript':
        normalized_code = transform_computer.to_mutable_tree(code_wrapped)
        normalized_code = JavaScriptStringifier().stringify(normalized_code)
        code_wrapped = normalized_code

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
        import traceback
        traceback.print_exc()
        print(type(e).__name__, e)
        for arg in job_args:
            print(arg)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 1:
        print('usage: python collect_feasible_transforms_jsonl.py <dataset>')

    DATASET = args[0]
    if DATASET in {'csn_java', 'github_java_funcs', 'mbjp'}:
        LANG = 'java'
    elif DATASET in {'github_c_funcs', 'mbcpp'}:
        LANG = 'cpp'
    elif DATASET in {'csn_js', 'mbjsp'}:
        LANG = 'javascript'
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
        ast_transformers.InfiniteLoopTransformer(),
        ast_transformers.UpdateTransformer(),
        ast_transformers.SameTypeDeclarationTransformer(),
        ast_transformers.VarDeclLocationTransformer(),
        ast_transformers.VarInitTransformer(),
        ast_transformers.VarNameStyleTransformer()
    ]

    data_processor = JsonlWMDatasetProcessor(LANG)
    if DATASET in {'mbjsp', 'mbjp', 'mbcpp'}:
        all_instances = data_processor._load_jsonl_fast(DATA_DIR, split='test')
    else:
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
                    jobs.append((LANG, instance.id, instance.source, key))

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

        feasible_transforms_per_file = dict()
        for iid, instance_dict in transforms_per_file.items():
            instance_dict_copy = dict(instance_dict)
            for t in code_transformers:
                transformer_name = t.name
                theoreticals = t.get_available_transforms()

                feasibles = list(instance_dict[transformer_name])
                if len(feasibles) < len(theoreticals):
                    # at least one transform is infeasible
                    for tt in theoreticals:
                        if tt not in feasibles:
                            # add first transform as a default value
                            feasibles.append(tt)
                            break

                instance_dict_copy[transformer_name] = feasibles

            def dfs_transform_compose(idict):
                all_feasible_transforms = []

                def _dfs_transform_compose(idict, i, cur_transforms):
                    if i == len(code_transformers):
                        all_feasible_transforms.append(cur_transforms)
                        return

                    transformer_name = code_transformers[i].name
                    for transform in idict[transformer_name]:
                        _dfs_transform_compose(idict, i + 1, cur_transforms + [transform])

                _dfs_transform_compose(idict, 0, [])
                return all_feasible_transforms

            all_feasible_transforms = dfs_transform_compose(instance_dict_copy)
            feasible_transforms_per_file[iid] = all_feasible_transforms

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
