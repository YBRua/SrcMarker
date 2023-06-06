from enum import Enum
from typing import List

from ..node import Node, NodeType
from .expression import Expression
from .expression import is_expression


class UnaryOps(Enum):
    PLUS = '+'
    NEG = '-'
    NOT = '!'
    BITWISE_NOT = '~'


_unary_op_map = {
    '+': UnaryOps.PLUS,
    '-': UnaryOps.NEG,
    '!': UnaryOps.NOT,
    '~': UnaryOps.BITWISE_NOT
}


def get_unary_op(op: str) -> UnaryOps:
    return _unary_op_map[op]


class UnaryExpression(Expression):

    def __init__(self, node_type: NodeType, operand: Expression, op: UnaryOps):
        super().__init__(node_type)
        self.operand = operand
        self.op = op
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.UNARY_EXPR:
            raise TypeError(f'Invalid type: {self.node_type} for UnaryExpression')
        if not is_expression(self.operand):
            raise TypeError(f'Invalid type: {self.operand.node_type} for unary operand')

    def to_string(self) -> str:
        return f'{self.op.value}{self.operand.to_string()}'

    def get_children(self) -> List[Node]:
        return [self.operand]

    def get_children_names(self) -> List[str]:
        return ['operand']
