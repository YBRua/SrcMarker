# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import treelib
import tree_sitter
from tree_sitter import Language, Parser

from .parser import (
    DFG_python,
    DFG_java,
    DFG_ruby,
    DFG_go,
    DFG_php,
    DFG_javascript,
    DFG_csharp,
)
from .parser import remove_comments_and_docstrings

root_dir = os.path.dirname(__file__)
dfg_function = {
    "c": DFG_java,
    "cpp": DFG_java,
    "python": DFG_python,
    "java": DFG_java,
    "ruby": DFG_ruby,
    "go": DFG_go,
    "php": DFG_php,
    "javascript": DFG_javascript,
    "c_sharp": DFG_csharp,
}


def check_tree_validity(root: tree_sitter.Node, max_depth: int = -1):
    return not root.has_error

    def _check_tree(node: tree_sitter.Node, depth: int = 0, max_depth: int = -1):
        valid = True
        if max_depth > 0 and depth > max_depth:
            return True
        if node.type == "ERROR":
            return False
        for child in node.children:
            valid = valid and _check_tree(child, depth + 1, max_depth=max_depth)
        return valid

    return _check_tree(root, depth=0, max_depth=max_depth)


def pprint_tree(root: tree_sitter.Node):
    tree = treelib.Tree()

    def _build_treelib_tree(current: tree_sitter.Node, parent=None):
        def _format_node(node: tree_sitter.Node):
            node_text = node.text.decode()
            if node.child_count == 0:
                node_str = f"{node.type} ({node_text})"
            else:
                node_str = f"{node.type}"
            # if node.type == 'identifier':
            #     node_str = f'{node_str} ({str(node.text, "utf-8")})'
            return node_str

        tree.create_node(_format_node(current), current.id, parent=parent)
        for child in current.children:
            _build_treelib_tree(child, current.id)

    _build_treelib_tree(root)
    tree.show(key=lambda x: True)  # keep order of insertion


def calc_syntax_match(references, candidate, lang):
    return corpus_syntax_match([references], [candidate], lang)


def corpus_syntax_match(references, candidates, lang):
    # track the compiled parser under parent directory
    JAVA_LANGUAGE = Language(os.path.join(root_dir, "..", "parser", "languages.so"), lang)
    parser = Parser()
    parser.set_language(JAVA_LANGUAGE)
    match_count = 0
    total_count = 0

    for i in range(len(candidates)):
        references_sample = references[i]
        candidate = candidates[i]
        for reference in references_sample:
            try:
                candidate = remove_comments_and_docstrings(candidate, "java")
            except:
                pass
            try:
                reference = remove_comments_and_docstrings(reference, "java")
            except:
                pass

            candidate_tree = parser.parse(bytes(candidate, "utf8")).root_node
            reference_tree = parser.parse(bytes(reference, "utf8")).root_node

            def get_all_sub_trees(root_node):
                node_stack = []
                sub_tree_sexp_list = []
                depth = 1
                node_stack.append([root_node, depth])
                while len(node_stack) != 0:
                    cur_node, cur_depth = node_stack.pop()
                    sub_tree_sexp_list.append([cur_node.sexp(), cur_depth])
                    for child_node in cur_node.children:
                        if len(child_node.children) != 0:
                            depth = cur_depth + 1
                            node_stack.append([child_node, depth])
                return sub_tree_sexp_list

            cand_sexps = [x[0] for x in get_all_sub_trees(candidate_tree)]
            ref_sexps = get_all_sub_trees(reference_tree)

            # print(cand_sexps)
            # print(ref_sexps)

            for sub_tree, depth in ref_sexps:
                if sub_tree in cand_sexps:
                    match_count += 1
            total_count += len(ref_sexps)

    score = match_count / total_count
    return score
