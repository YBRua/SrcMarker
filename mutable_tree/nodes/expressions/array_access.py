from typing import List
from mutable_tree.nodes.node import Node
from ..node import NodeType
from .expression import Expression
from .expression import is_primary_expression, is_expression


class ArrayAccess(Expression):
    def __init__(
        self,
        node_type: NodeType,
        array: Expression,
        index: Expression,
        optional: bool = False,
    ):
        super().__init__(node_type)
        self.array = array
        self.index = index
        self.optional = optional
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.ARRAY_ACCESS:
            raise TypeError(f"Invalid type: {self.node_type} for ArrayAccess")
        if not is_primary_expression(self.array):
            raise TypeError(f"Invalid type: {self.array.node_type} for array")
        if not is_expression(self.index):
            raise TypeError(f"Invalid type: {self.index.node_type} for array index")

    def get_children(self) -> List[Node]:
        return [self.array, self.index]

    def get_children_names(self) -> List[str]:
        return ["array", "index"]
