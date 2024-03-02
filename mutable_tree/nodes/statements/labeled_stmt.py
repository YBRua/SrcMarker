from ..node import Node, NodeType
from .statement import Statement
from .statement import is_statement
from ..expressions import Identifier
from ..utils import throw_invalid_type
from typing import List


class LabeledStatement(Statement):
    def __init__(self, node_type: NodeType, label: Identifier, stmt: Statement):
        super().__init__(node_type)
        self.label = label
        self.stmt = stmt
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.LABELED_STMT:
            throw_invalid_type(self.node_type, self)
        if self.label.node_type != NodeType.IDENTIFIER:
            throw_invalid_type(self.label.node_type, self, attr="label")
        if not is_statement(self.stmt):
            throw_invalid_type(self.stmt.node_type, self, attr="stmt")

    def get_children(self) -> List[Node]:
        return [self.label, self.stmt]

    def get_children_names(self) -> List[str]:
        return ["label", "stmt"]
