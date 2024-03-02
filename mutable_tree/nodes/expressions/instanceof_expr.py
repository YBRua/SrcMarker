from typing import List
from mutable_tree.nodes.node import Node
from ..node import NodeType
from .expression import Expression
from ..types import TypeIdentifier
from .expression import is_expression


class InstanceofExpression(Expression):
    def __init__(self, node_type: NodeType, left: Expression, right: TypeIdentifier):
        super().__init__(node_type)
        self.left = left
        self.right = right
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.INSTANCEOF_EXPR:
            raise TypeError(f"Invalid type: {self.node_type} for InstanceofExpression")
        if not is_expression(self.left):
            raise TypeError(f"Invalid type: {self.left.node_type} for instanceof LHS")
        if self.right.node_type != NodeType.TYPE_IDENTIFIER:
            raise TypeError(f"Invalid type: {self.right.node_type} for instanceof RHS")

    def get_children(self) -> List[Node]:
        return [self.left, self.right]

    def get_children_names(self) -> List[str]:
        return ["left", "right"]
