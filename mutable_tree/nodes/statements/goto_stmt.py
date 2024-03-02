from ..node import Node, NodeType
from .statement import Statement
from ..expressions import Identifier
from ..utils import throw_invalid_type
from typing import List


class GotoStatement(Statement):
    def __init__(self, node_type: NodeType, label: Identifier):
        super().__init__(node_type)
        self.label = label
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.GOTO_STMT:
            throw_invalid_type(self.node_type, self)
        if self.label.node_type != NodeType.IDENTIFIER:
            throw_invalid_type(self.label.node_type, self, attr="label")

    def get_children(self) -> List[Node]:
        return [self.label]

    def get_children_names(self) -> List[str]:
        return ["label"]
