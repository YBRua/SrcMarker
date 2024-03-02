from ..node import Node, NodeType
from ..utils import throw_invalid_type
from .expression import Expression
from .expression import is_expression
from typing import List


class SpreadElement(Expression):
    def __init__(self, node_type: NodeType, expr: Expression):
        super().__init__(node_type)
        self.expr = expr
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.SPREAD_ELEMENT:
            throw_invalid_type(self.node_type, self)
        if not is_expression(self.expr):
            throw_invalid_type(self.expr.node_type, self, attr="expr")

    def get_children(self) -> List[Node]:
        return [self.expr]

    def get_children_names(self) -> List[str]:
        return ["expr"]
