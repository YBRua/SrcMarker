from .expression import Expression
from ..node import Node, NodeType
from typing import List


class Identifier(Expression):
    def __init__(self, node_type: NodeType, name: str):
        super().__init__(node_type)
        self.name = name
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.IDENTIFIER:
            raise TypeError(f"Invalid type: {self.node_type} for Identifier")

    def get_children(self) -> List[Node]:
        return []

    def get_children_names(self) -> List[str]:
        return []
