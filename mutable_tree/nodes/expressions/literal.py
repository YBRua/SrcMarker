from .expression import Expression
from ..node import NodeType
from typing import List, Union


class Literal(Expression):
    def __init__(self, node_type: NodeType, value: Union[str, int, float]):
        super().__init__(node_type)
        self.value = value
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.LITERAL:
            raise TypeError(f"Invalid type: {self.node_type} for Literal")

    def get_children(self):
        return []

    def get_children_names(self) -> List[str]:
        return []
