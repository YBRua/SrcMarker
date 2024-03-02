from ..node import Node, NodeType
from ..types import TypeIdentifier
from ..utils import throw_invalid_type
from .expression import Expression
from .expression import is_expression
from typing import List, Union


class SizeofExpression(Expression):
    def __init__(self, node_type: NodeType, operand: Union[Expression, TypeIdentifier]):
        super().__init__(node_type)
        self.operand = operand
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.SIZEOF_EXPR:
            throw_invalid_type(self.node_type, self)
        if (
            not is_expression(self.operand)
            and self.operand.node_type != NodeType.TYPE_IDENTIFIER
        ):
            throw_invalid_type(self.operand.node_type, self, attr="operand")

    def get_children(self) -> List[Node]:
        return [self.operand]

    def get_children_names(self) -> List[str]:
        return ["operand"]
