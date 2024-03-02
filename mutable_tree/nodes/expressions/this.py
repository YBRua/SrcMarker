from ..node import Node, NodeType
from .expression import Expression
from typing import List


class ThisExpression(Expression):
    def __init__(self, node_type: NodeType):
        super().__init__(node_type)
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.THIS_EXPR:
            raise TypeError(f"Invalid type: {self.node_type} for ThisExpression")

    def get_children(self) -> List[Node]:
        return []

    def get_children_names(self) -> List[str]:
        return []
