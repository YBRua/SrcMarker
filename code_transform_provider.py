import tree_sitter
from itertools import product
from mutable_tree.nodes import Node
from mutable_tree.adaptors import JavaAdaptor, CppAdaptor
from mutable_tree.transformers import CodeTransformer, TransformerPipeline
from mutable_tree.tree_manip.visitors import IdentifierRenamingVisitor

from typing import Sequence, List


class CodeTransformProvider:
    def __init__(self, lang: str, parser: tree_sitter.Parser,
                 transformers: List[CodeTransformer]) -> None:
        self.lang = lang
        self.parser = parser
        self.pipeline = TransformerPipeline(transformers)
        self.transform_keys = self._get_all_transform_combinations(transformers)

        if self.lang not in {'cpp', 'java'}:
            raise ValueError(f'lang must be one of "cpp" or "java", got {lang}')

    def _get_all_transform_combinations(self,
                                        transformers: List[CodeTransformer]) -> List[str]:
        all_keys = []
        for t in transformers:
            keys = t.get_available_transforms()
            all_keys.append(keys)
        return list(product(*all_keys))

    def _wrap_code(self, code: str):
        if self.lang == 'java':
            return f'public class A {{ {code} }}'
        else:
            return code

    def _get_java_function_root(self, root: tree_sitter.Node) -> tree_sitter.Node:
        assert root.type == 'program'
        class_decl_node = root.children[0]
        assert class_decl_node.type == 'class_declaration'
        class_body_node = class_decl_node.children[3]
        assert class_body_node.type == 'class_body'
        func_root_node = class_body_node.children[1]
        assert func_root_node.type == 'method_declaration', func_root_node.type
        return func_root_node

    def _get_cpp_function_root(self, root: tree_sitter.Node) -> tree_sitter.Node:
        assert root.type == 'translation_unit'
        func_root_node = root.children[0]
        assert func_root_node.type == 'function_definition'
        return func_root_node

    def _get_function_root(self, root: tree_sitter.Node) -> tree_sitter.Node:
        if self.lang == 'java':
            return self._get_java_function_root(root)
        else:
            return self._get_cpp_function_root(root)

    def to_mutable_tree(self, code: str) -> Node:
        code = self._wrap_code(code)
        tree = self.parser.parse(bytes(code, 'utf-8'))
        func_root = self._get_function_root(tree.root_node)
        if self.lang == 'java':
            return JavaAdaptor.convert_function_declaration(func_root)
        else:
            return CppAdaptor.convert_function_definition(func_root)

    def variable_substitution(self, code: str, src_var: str, dst_var: str) -> str:
        try:
            mutable_root = self.to_mutable_tree(code)
        except Exception as e:
            print(f'Failed to parse code: {code}')
            raise e
        visitor = IdentifierRenamingVisitor(src_var, dst_var)
        new_root = visitor.visit(mutable_root)
        return new_root.to_string()

    def code_transform(self, code: str, transform_keys: Sequence[str]) -> str:
        mutable_root = self.to_mutable_tree(code)
        transformed = self.pipeline.mutable_tree_transform(mutable_root, transform_keys)
        return transformed.to_string()

    def get_transform_keys(self) -> List[str]:
        return self.transform_keys
