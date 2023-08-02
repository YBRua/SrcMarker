import tree_sitter
from itertools import product
from mutable_tree.nodes import Node
from mutable_tree.adaptors import JavaAdaptor, CppAdaptor, JavaScriptAdaptor
from mutable_tree.stringifiers import (JavaStringifier, CppStringifier,
                                       JavaScriptStringifier)
from mutable_tree.transformers import CodeTransformer, TransformerPipeline
from mutable_tree.tree_manip.visitors import (IdentifierRenamingVisitor,
                                              IdentifierAppendingVisitor)

from typing import Sequence, List


class CodeTransformProvider:
    def __init__(self, lang: str, parser: tree_sitter.Parser,
                 transformers: List[CodeTransformer]) -> None:
        self.lang = lang
        self.parser = parser
        self.pipeline = TransformerPipeline(transformers)
        self.transform_keys = self._get_all_transform_combinations(transformers)

        if self.lang not in {'cpp', 'java', 'javascript'}:
            raise ValueError(f'Unsupported language {lang}')

        if self.lang == 'java':
            self.stringifier = JavaStringifier()
        elif self.lang == 'cpp':
            self.stringifier = CppStringifier()
        elif self.lang == 'javascript':
            self.stringifier = JavaScriptStringifier()

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

    def _get_js_function_root(self, root: tree_sitter.Node) -> tree_sitter.Node:
        assert root.type == 'program'
        func_root_node = root.children[0]

        if func_root_node.type == 'function_declaration':
            return func_root_node
        elif func_root_node.type == 'expression_statement':
            func_root_node = func_root_node.children[0]
            assert func_root_node.type == 'function', func_root_node.type
            return func_root_node
        elif func_root_node.type == 'generator_function_declaration':
            return func_root_node
        else:
            raise RuntimeError(f'Unexpected root node type: {func_root_node.type}')

    def _get_function_root(self, root: tree_sitter.Node) -> tree_sitter.Node:
        if self.lang == 'java':
            return self._get_java_function_root(root)
        elif self.lang == 'javascript':
            return self._get_js_function_root(root)
        else:
            assert self.lang == 'cpp'
            return self._get_cpp_function_root(root)

    def to_mutable_tree(self, code: str) -> Node:
        code = self._wrap_code(code)
        tree = self.parser.parse(bytes(code, 'utf-8'))
        func_root = self._get_function_root(tree.root_node)
        if self.lang == 'java':
            return JavaAdaptor.convert_function_declaration(func_root)
        elif self.lang == 'cpp':
            return CppAdaptor.convert_function_definition(func_root)
        elif self.lang == 'javascript':
            return JavaScriptAdaptor.convert_function_declaration(func_root)
        else:
            raise RuntimeError('Unreachable')

    def variable_append(self, code: str, src_var: str, append_var: str) -> str:
        try:
            mutable_root = self.to_mutable_tree(code)
        except Exception as e:
            print(f'Failed to parse code: {code}, msg: {e}')
            return code
        visitor = IdentifierAppendingVisitor(src_var, append_var)
        new_root = visitor.visit(mutable_root)
        return self.stringifier.stringify(new_root)

    def variable_substitution(self, code: str, src_var: str, dst_var: str) -> str:
        try:
            mutable_root = self.to_mutable_tree(code)
        except Exception as e:
            print(f'Failed to parse code: {code}, msg: {e}')
            return code
        visitor = IdentifierRenamingVisitor(src_var, dst_var)
        new_root = visitor.visit(mutable_root)
        return self.stringifier.stringify(new_root)

    def code_transform(self, code: str, transform_keys: Sequence[str]) -> str:
        mutable_root = self.to_mutable_tree(code)
        transformed = self.pipeline.mutable_tree_transform(mutable_root, transform_keys)
        return self.stringifier.stringify(transformed)

    def get_transform_keys(self) -> List[str]:
        return self.transform_keys
