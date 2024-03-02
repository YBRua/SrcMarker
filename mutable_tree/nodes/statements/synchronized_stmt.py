from ..node import Node, NodeType
from .statement import Statement
from .block_stmt import BlockStatement
from ..expressions import ParenthesizedExpression
from ..utils import throw_invalid_type
from typing import List


class SynchronizedStatement(Statement):
    def __init__(
        self, node_type: NodeType, expr: ParenthesizedExpression, body: BlockStatement
    ):
        super().__init__(node_type)
        self.expr = expr
        self.body = body
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.SYNCHRONIZED_STMT:
            throw_invalid_type(self.node_type, self)
        if self.expr.node_type != NodeType.PARENTHESIZED_EXPR:
            throw_invalid_type(self.expr.node_type, self, attr="expr")
        if self.body.node_type != NodeType.BLOCK_STMT:
            throw_invalid_type(self.body.node_type, self, attr="body")

    def get_children(self) -> List[Node]:
        return [self.expr, self.body]

    def get_children_names(self) -> List[str]:
        return ["expr", "body"]
