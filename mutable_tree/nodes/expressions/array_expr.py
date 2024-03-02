from ..node import Node, NodeType
from .expression import Expression
from .expression_list import ExpressionList
from ..utils import throw_invalid_type
from typing import List


class ArrayExpression(Expression):
    def __init__(self, node_type: NodeType, elements: ExpressionList):
        super().__init__(node_type)
        self.elements = elements
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.ARRAY_EXPR:
            throw_invalid_type(self.node_type, self)
        if self.elements.node_type != NodeType.EXPRESSION_LIST:
            throw_invalid_type(self.elements.node_type, self, "elements")

    def get_children(self) -> List[Node]:
        return [self.elements]

    def get_children_names(self) -> List[str]:
        return ["elements"]
