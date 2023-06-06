from .visitor import StatefulTransformingVisitor
from .var_name_utils import normalize_name

from mutable_tree.nodes import Node
from mutable_tree.nodes import node_factory
from mutable_tree.nodes import Identifier
from typing import Optional


class IdentifierRenamingVisitor(StatefulTransformingVisitor):

    def __init__(self, src_var: str, dst_var: str) -> None:
        super().__init__()
        self.src_var = src_var
        self.dst_var = dst_var

        self.norm_src = normalize_name(src_var).lower()
        self.norm_dst = normalize_name(dst_var).lower()

    def visit_Identifier(self,
                         node: Identifier,
                         parent: Optional[Node] = None,
                         parent_attr: Optional[str] = None):
        normalized_name = normalize_name(node.name).lower()
        if normalized_name == self.norm_src:
            new_id = node_factory.create_identifier(self.dst_var)
            return True, [new_id]
        else:
            return False, []
