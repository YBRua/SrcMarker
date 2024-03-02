from ..node import Node, NodeType
from .statement import Statement
from ..expressions import Expression
from ..expressions import is_expression
from ..utils import throw_invalid_type
from typing import List, Optional


class YieldStatement(Statement):
    def __init__(
        self,
        node_type: NodeType,
        expr: Optional[Expression] = None,
        is_delegate: bool = False,
    ):
        super().__init__(node_type)
        self.expr = expr
        self.is_delegate = is_delegate
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.YIELD_STMT:
            throw_invalid_type(self.node_type, self)
        if self.expr is not None and not is_expression(self.expr):
            throw_invalid_type(self.expr.node_type, self, attr="expr")

    def get_children(self) -> List[Node]:
        return [self.expr]

    def get_children_names(self) -> List[str]:
        return ["expr"]
