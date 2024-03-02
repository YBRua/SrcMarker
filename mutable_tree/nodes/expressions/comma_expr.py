from ..node import Node, NodeType
from ..utils import throw_invalid_type
from .expression import Expression
from .expression import is_expression
from typing import List


class CommaExpression(Expression):
    def __init__(self, node_type: NodeType, left: Expression, right: Expression):
        super().__init__(node_type)
        self.left = left
        self.right = right
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.COMMA_EXPR:
            throw_invalid_type(self.node_type, self)
        if not is_expression(self.left):
            throw_invalid_type(self.left.node_type, self, attr="left")
        if not is_expression(self.right):
            throw_invalid_type(self.right.node_type, self, attr="right")

    def get_children(self) -> List[Node]:
        return [self.left, self.right]

    def get_children_names(self) -> List[str]:
        return ["left", "right"]
