from ..node import Node, NodeType
from .statement import Statement
from ..utils import throw_invalid_type
from typing import List


class EmptyStatement(Statement):
    def __init__(self, node_type: NodeType):
        super().__init__(node_type)
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.EMPTY_STMT:
            throw_invalid_type(self.node_type, self)

    def get_children(self) -> List[Node]:
        return []

    def get_children_names(self) -> List[str]:
        return []
