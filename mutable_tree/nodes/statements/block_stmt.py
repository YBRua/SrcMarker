from ..node import Node, NodeType
from .statement import Statement
from .statement_list import StatementList
from ..utils import throw_invalid_type
from typing import List


class BlockStatement(Statement):
    def __init__(self, node_type: NodeType, stmts: StatementList):
        super().__init__(node_type)
        self.stmts = stmts
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.BLOCK_STMT:
            throw_invalid_type(self.node_type, self)
        if self.stmts.node_type != NodeType.STATEMENT_LIST:
            throw_invalid_type(self.stmts.node_type, self, "stmts")

    def get_children(self) -> List[Node]:
        return [self.stmts]

    def get_children_names(self) -> List[str]:
        return ["stmts"]
