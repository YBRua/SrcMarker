from ..node import Node, NodeType
from .statement import Statement
from ..expressions import Expression
from typing import List, Optional


class ContinueStatement(Statement):
    def __init__(self, node_type: NodeType, label: Optional[Expression] = None):
        super().__init__(node_type)
        self.label = label
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.CONTINUE_STMT:
            raise TypeError(f"Invalid type: {self.node_type} for ContinueStatement")
        if self.label is not None and self.label.node_type != NodeType.IDENTIFIER:
            raise TypeError(f"Invalid type: {self.label.node_type} for continue label")

    def get_children(self) -> List[Node]:
        if self.label is not None:
            return [self.label]
        else:
            return []

    def get_children_names(self) -> List[str]:
        return ["label"]
