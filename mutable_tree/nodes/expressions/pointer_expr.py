from enum import Enum
from ..node import Node, NodeType
from ..utils import throw_invalid_type
from .expression import Expression
from .expression import is_expression
from typing import List


class PointerOps(Enum):
    DEREF = "*"
    ADDRESS = "&"


_pointer_op_map = {"*": PointerOps.DEREF, "&": PointerOps.ADDRESS}


def get_pointer_op(op: str) -> PointerOps:
    return _pointer_op_map[op]


class PointerExpression(Expression):
    def __init__(self, node_type: NodeType, operand: Expression, op: PointerOps):
        super().__init__(node_type)
        self.operand = operand
        self.op = op
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.POINTER_EXPR:
            throw_invalid_type(self.node_type, self)
        if not is_expression(self.operand):
            throw_invalid_type(self.operand.node_type, self, attr="operand")

    def get_children(self) -> List[Node]:
        return [self.operand]

    def get_children_names(self) -> List[str]:
        return ["operand"]
