from ..node import Node, NodeType
from .statement import Statement
from ..expressions import Expression, is_expression
from ..utils import throw_invalid_type
from typing import List, Optional


class ReturnStatement(Statement):
    def __init__(self, node_type: NodeType, expr: Optional[Expression] = None):
        super().__init__(node_type)
        self.expr = expr
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.RETURN_STMT:
            throw_invalid_type(self.node_type, self)
        if (
            self.expr is not None
            and not is_expression(self.expr)
            and self.expr.node_type != NodeType.FUNCTION_DEFINITION
        ):
            throw_invalid_type(self.expr.node_type, self, attr="expr")

    def get_children(self) -> List[Node]:
        if self.expr is not None:
            return [self.expr]
        else:
            return []

    def get_children_names(self) -> List[str]:
        return ["expr"]
