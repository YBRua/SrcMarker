from .visitor import StatefulTransformingVisitor
from .var_name_utils import normalize_name

from mutable_tree.nodes import Node, NodeType
from mutable_tree.nodes import node_factory
from mutable_tree.nodes import Identifier, FunctionDeclarator
from typing import Optional


class IdentifierAppendingVisitor(StatefulTransformingVisitor):
    def __init__(self, src_var: str, dst_var: str) -> None:
        super().__init__()
        self.src_var = src_var
        self.dst_var = dst_var

        self.norm_src = normalize_name(src_var).lower()
        self.norm_dst = normalize_name(dst_var).lower()

    def visit_FunctionDeclarator(
        self,
        node: FunctionDeclarator,
        parent: Optional[Node] = None,
        parent_attr: Optional[str] = None,
    ):
        if parent.node_type == NodeType.FUNCTION_HEADER:
            self.generic_visit(node.parameters, node, "parameters")
            return False, []
        else:
            return self.generic_visit(node, parent, parent_attr)

    def visit_Identifier(
        self,
        node: Identifier,
        parent: Optional[Node] = None,
        parent_attr: Optional[str] = None,
    ):
        if parent.node_type == NodeType.FIELD_ACCESS and parent_attr == "field":
            return False, []

        if parent.node_type == NodeType.CALL_EXPR and parent_attr == "callee":
            return False, []

        normalized_name = normalize_name(node.name).lower()
        if normalized_name == self.norm_src:
            if len(self.dst_var) == 1:
                new_name = self.src_var + self.dst_var[0].upper()
            else:
                new_name = self.src_var + self.dst_var[0].upper() + self.dst_var[1:]
            new_id = node_factory.create_identifier(new_name)
            return True, [new_id]
        else:
            return False, []


class IdentifierRenamingVisitor(StatefulTransformingVisitor):
    def __init__(self, src_var: str, dst_var: str) -> None:
        super().__init__()
        self.src_var = src_var
        self.dst_var = dst_var

        self.norm_src = normalize_name(src_var).lower()
        self.norm_dst = normalize_name(dst_var).lower()

    def visit_FunctionDeclarator(
        self,
        node: FunctionDeclarator,
        parent: Optional[Node] = None,
        parent_attr: Optional[str] = None,
    ):
        if parent.node_type == NodeType.FUNCTION_HEADER:
            self.generic_visit(node.parameters, node, "parameters")
            return False, []
        else:
            return self.generic_visit(node, parent, parent_attr)

    def visit_Identifier(
        self,
        node: Identifier,
        parent: Optional[Node] = None,
        parent_attr: Optional[str] = None,
    ):
        if parent.node_type == NodeType.FIELD_ACCESS and parent_attr == "field":
            return False, []

        if parent.node_type == NodeType.CALL_EXPR and parent_attr == "callee":
            return False, []

        normalized_name = normalize_name(node.name).lower()
        if normalized_name == self.norm_src:
            new_id = node_factory.create_identifier(self.dst_var)
            return True, [new_id]
        else:
            return False, []
