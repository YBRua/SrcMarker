from enum import Enum
from typing import List

from mutable_tree.nodes.node import Node
from ..node import NodeType
from .expression import Expression
from .expression import is_expression


class BinaryOps(Enum):
    PLUS = '+'
    MINUS = '-'
    MULTIPLY = '*'
    DIVIDE = '/'
    GT = '>'
    LT = '<'
    GE = '>='
    LE = '<='
    EQ = '=='
    NE = '!='
    AND = '&&'
    OR = '||'
    BITWISE_XOR = '^'
    BITWISE_AND = '&'
    BITWISE_OR = '|'
    MOD = '%'
    LSHIFT = '<<'
    RSHIFT = '>>'
    LRSHIFT = '>>>'


_binary_op_map = {
    '+': BinaryOps.PLUS,
    '-': BinaryOps.MINUS,
    '*': BinaryOps.MULTIPLY,
    '/': BinaryOps.DIVIDE,
    '>': BinaryOps.GT,
    '<': BinaryOps.LT,
    '>=': BinaryOps.GE,
    '<=': BinaryOps.LE,
    '==': BinaryOps.EQ,
    '!=': BinaryOps.NE,
    '&&': BinaryOps.AND,
    '||': BinaryOps.OR,
    '^': BinaryOps.BITWISE_XOR,
    '&': BinaryOps.BITWISE_AND,
    '|': BinaryOps.BITWISE_OR,
    '%': BinaryOps.MOD,
    '<<': BinaryOps.LSHIFT,
    '>>': BinaryOps.RSHIFT,
    '>>>': BinaryOps.LRSHIFT
}


def get_binary_op(op: str) -> BinaryOps:
    return _binary_op_map[op]


class BinaryExpression(Expression):

    def __init__(self, node_type: NodeType, left: Expression, right: Expression,
                 op: BinaryOps):
        super().__init__(node_type)
        self.left = left
        self.right = right
        self.op = op
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.BINARY_EXPR:
            raise TypeError(f'Invalid type: {self.node_type} for BinaryExpression')
        if not is_expression(self.left):
            raise TypeError(f'Invalid type: {self.left.node_type} for BinOp LHS')
        if not is_expression(self.right):
            raise TypeError(f'Invalid type: {self.right.node_type} for BinOp RHS')

    def to_string(self) -> str:
        return f'{self.left.to_string()} {self.op.value} {self.right.to_string()}'

    def get_children(self) -> List[Node]:
        return [self.left, self.right]

    def get_children_names(self) -> List[str]:
        return ['left', 'right']
