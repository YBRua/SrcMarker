import sys
import json
import pprint
import tree_sitter
from tqdm import tqdm
from collections import Counter
from data_processing import JsonlWMDatasetProcessor
from code_transform_provider import CodeTransformProvider
from mutable_tree.nodes import (
    Node,
    Identifier,
    LocalVariableDeclaration,
    FunctionDeclarator,
    Declarator,
    VariableDeclarator,
    CallExpression,
    FunctionHeader,
    FieldAccess,
)
from typing import List


def variable_collector(node: Node) -> List[str]:
    variable_names: List[str] = []

    def _find_identifier_from_func_declarator(node: Declarator):
        if isinstance(node, FunctionDeclarator):
            _find_identifier(node.parameters)
        elif isinstance(node, VariableDeclarator):
            return
        else:
            _find_identifier(node.declarator)

    def _find_identifier_from_field_access(node: FieldAccess):
        _find_identifier(node.object)

    def _find_identifier(node: Node):
        if isinstance(node, Identifier):
            variable_names.append(node.name)
        elif isinstance(node, CallExpression):
            return
        elif isinstance(node, FieldAccess):
            _find_identifier_from_field_access(node)
        elif isinstance(node, FunctionHeader):
            _find_identifier_from_func_declarator(node.func_decl)
        else:
            for child_name in node.get_children_names():
                child = node.get_child_at(child_name)
                if child is not None:
                    _find_identifier(child)

    def _collect_variable_declarator(node: Declarator):
        if isinstance(node, VariableDeclarator):
            _find_identifier(node)
        else:
            _collect_variable_declarator(node.declarator)

    def _collect_local_variable_declaration(node: LocalVariableDeclaration):
        declarators = node.declarators
        for decl_attr in declarators.get_children_names():
            decl = declarators.get_child_at(decl_attr)
            if decl is not None:
                _collect_variable_declarator(decl)

    def _collect_formal_parameters(node: FunctionDeclarator):
        formal_params = node.parameters
        for param_attr in formal_params.get_children_names():
            param = formal_params.get_child_at(param_attr)
            if param is not None:
                _find_identifier(param)

    def _variable_collector(node: Node):
        for child_attr in node.get_children_names():
            child = node.get_child_at(child_attr)
            if child is not None:
                _find_identifier(child)

    _variable_collector(node)
    return variable_names


def main(args):
    if len(args) != 1:
        print("Usage: python collect_variable_names_jsonl.py <dataset>")
        return

    DATASET = args[0]
    if DATASET in {"csn_java", "github_java_funcs", "mbjp"}:
        LANG = "java"
    elif DATASET in {"github_c_funcs", "mbcpp"}:
        LANG = "cpp"
    elif DATASET in {"csn_js", "mbjsp"}:
        LANG = "javascript"
    else:
        raise ValueError(f"Unknown dataset: {DATASET}")

    DATA_DIR = f"./datasets/{DATASET}"

    parser = tree_sitter.Parser()
    parser_lang = tree_sitter.Language("./parser/languages.so", LANG)
    parser.set_language(parser_lang)
    transform_computer = CodeTransformProvider(LANG, parser, [])

    data_processor = JsonlWMDatasetProcessor(LANG)
    if DATASET in {"mbjsp", "mbjp", "mbcpp"}:
        all_instances = data_processor._load_jsonl_fast(DATA_DIR, split="test")
    else:
        instances = data_processor.load_jsonls_fast(DATA_DIR, show_progress=False)
        train_instances = instances["train"]
        valid_instances = instances["valid"]
        test_instances = instances["test"]

        all_instances = train_instances + valid_instances + test_instances

    all_variable_names = set()
    variable_names_per_file = dict()

    for instance in tqdm(all_instances):
        code = instance.source
        mutable_root = transform_computer.to_mutable_tree(code)
        variable_names = variable_collector(mutable_root)
        variable_names_per_file[instance.id] = dict(Counter(variable_names))
        all_variable_names.update(variable_names)

    res_dict = {
        "all_variable_names": list(all_variable_names),
        "variable_names_per_file": variable_names_per_file,
    }

    json.dump(
        res_dict, open(f"./datasets/variable_names_{DATASET}.json", "w"), indent=2
    )
    variable_names_per_file = res_dict["variable_names_per_file"]

    var_counts = []
    tot_files = 0
    for instance_id, instance_list in variable_names_per_file.items():
        var_counts.append(len(instance_list))
        tot_files += 1

    var_counter = Counter(var_counts)
    pprint.pp(var_counter)

    tot_vars = sum(var_counts)
    print(f"total files: {tot_files}")
    print(f'total var names: {len(res_dict["all_variable_names"])}')
    print(f"total var count: {tot_vars}")
    print(f"avg vars per file: {tot_vars / tot_files:.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
