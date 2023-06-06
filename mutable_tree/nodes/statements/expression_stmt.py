from ..node import Node, NodeType
from .statement import Statement
from ..expressions import Expression
from ..expressions import is_expression
from ..utils import throw_invalid_type
from typing import List


class ExpressionStatement(Statement):

    def __init__(self, node_type: NodeType, expr: Expression):
        super().__init__(node_type)
        self.expr = expr
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.EXPRESSION_STMT:
            throw_invalid_type(self.node_type, self)
        if not is_expression(self.expr):
            throw_invalid_type(self.expr.node_type, self, attr='expr')

    def to_string(self) -> str:
        return f'{self.expr.to_string()};'

    def get_children(self) -> List[Node]:
        return [self.expr]

    def get_children_names(self) -> List[str]:
        return ['expr']
