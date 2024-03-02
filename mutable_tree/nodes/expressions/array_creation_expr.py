from ..node import Node, NodeType
from ..types import TypeIdentifier, Dimensions
from .expression import Expression
from .array_expr import ArrayExpression
from ..utils import throw_invalid_type
from typing import List, Optional


class ArrayCreationExpression(Expression):
    def __init__(
        self,
        node_type: NodeType,
        type: TypeIdentifier,
        dimensions: Dimensions,
        value: Optional[ArrayExpression] = None,
    ):
        super().__init__(node_type)
        self.type_id = type
        self.dimensions = dimensions
        self.value = value
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.ARRAY_CREATION_EXPR:
            throw_invalid_type(self.node_type, self)
        if self.type_id.node_type != NodeType.TYPE_IDENTIFIER:
            throw_invalid_type(self.type_id.node_type, self, "type")
        if self.dimensions.node_type != NodeType.DIMENSIONS:
            throw_invalid_type(self.dimensions.node_type, self, "dimensions")
        if self.value is not None and self.value.node_type != NodeType.ARRAY_EXPR:
            throw_invalid_type(self.value.node_type, self, "value")

    def get_children(self) -> List[Node]:
        res = [self.type_id, self.dimensions]
        if self.value is not None:
            res.append(self.value)
        return res

    def get_children_names(self) -> List[str]:
        return ["type_id", "dimensions", "value"]
