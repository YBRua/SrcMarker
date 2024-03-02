from enum import Enum
from ..node import Node, NodeType
from ..utils import throw_invalid_type
from .expression import Expression
from .expression import is_primary_expression
from typing import List


class FieldAccessOps(Enum):
    DOT = "."
    PTR = "->"


_field_access_op_map = {".": FieldAccessOps.DOT, "->": FieldAccessOps.PTR}


def get_field_access_op(op: str) -> FieldAccessOps:
    return _field_access_op_map[op]


class FieldAccess(Expression):
    def __init__(
        self,
        node_type: NodeType,
        object: Expression,
        field: Expression,
        op: FieldAccessOps = FieldAccessOps.DOT,
        optional: bool = False,
    ):
        super().__init__(node_type)
        self.object = object
        self.field = field
        self.op = op
        self.optional = optional
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.FIELD_ACCESS:
            throw_invalid_type(self.node_type, self)
        if (
            not is_primary_expression(self.object)
            and self.object.node_type != NodeType.ARRAY_EXPR
            and self.object.node_type != NodeType.FUNCTION_DEFINITION
        ):
            throw_invalid_type(self.object.node_type, self, attr="object")
        f_nt = self.field.node_type
        if f_nt != NodeType.IDENTIFIER and f_nt != NodeType.THIS_EXPR:
            throw_invalid_type(f_nt, self, attr="field")

    def get_children(self) -> List[Node]:
        return [self.object, self.field]

    def get_children_names(self) -> List[str]:
        return ["object", "field"]
