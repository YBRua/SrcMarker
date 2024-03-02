from ..node import Node, NodeType
from .statement import Statement
from .statement import is_statement
from ..expressions import Expression, is_expression
from ..utils import throw_invalid_type
from typing import List, Optional


class IfStatement(Statement):
    def __init__(
        self,
        node_type: NodeType,
        condition: Expression,
        consequence: Statement,
        alternate: Optional[Statement] = None,
    ):
        super().__init__(node_type)
        self.condition = condition
        self.consequence = consequence
        self.alternate = alternate
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.IF_STMT:
            throw_invalid_type(self.node_type, self)
        if not is_expression(self.condition):
            throw_invalid_type(self.condition.node_type, self, attr="condition")
        if not is_statement(self.consequence):
            throw_invalid_type(self.consequence.node_type, self, attr="consequence")
        if self.alternate is not None and not is_statement(self.alternate):
            throw_invalid_type(self.alternate.node_type, self, attr="alternate")

    def get_children(self) -> List[Node]:
        if self.alternate is None:
            return [self.condition, self.consequence]
        else:
            return [self.condition, self.consequence, self.alternate]

    def get_children_names(self) -> List[str]:
        return ["condition", "consequence", "alternate"]
