from ..node import Node, NodeType
from ..types import TypeIdentifier
from .expression import Expression
from .array_expr import ArrayExpression
from ..utils import throw_invalid_type
from typing import List


class CompoundLiteralExpression(Expression):
    def __init__(
        self, node_type: NodeType, type: TypeIdentifier, value: ArrayExpression
    ):
        super().__init__(node_type)
        self.type_id = type
        self.value = value
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.COMPOUND_LITERAL_EXPR:
            throw_invalid_type(self.node_type, self)
        if self.type_id.node_type != NodeType.TYPE_IDENTIFIER:
            throw_invalid_type(self.type_id.node_type, self, "type")
        if self.value.node_type != NodeType.ARRAY_EXPR:
            throw_invalid_type(self.value.node_type, self, "value")

    def get_children(self) -> List[Node]:
        return [self.type_id, self.value]

    def get_children_names(self) -> List[str]:
        return ["type_id", "value"]
