from ..node import Node, NodeType
from ..statements import StatementList
from ..utils import throw_invalid_type
from typing import List


class Program(Node):
    def __init__(self, node_type: NodeType, stmts: StatementList):
        super().__init__(node_type)
        self.main = stmts
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.PROGRAM:
            throw_invalid_type(self.node_type, self)
        if self.main.node_type != NodeType.STATEMENT_LIST:
            throw_invalid_type(self.main.node_type, self, "main")

    def get_children(self) -> List[Node]:
        return [self.main]

    def get_children_names(self) -> List[str]:
        return ["main"]
