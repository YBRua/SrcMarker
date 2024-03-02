from ..node import NodeType, NodeList
from .expression import Expression
from .expression import is_expression
from ..utils import throw_invalid_type
from typing import List


class ExpressionList(NodeList):
    node_list: List[Expression]

    def __init__(self, node_type: NodeType, exprs: List[Expression]):
        super().__init__(node_type)
        self.node_list = exprs
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.EXPRESSION_LIST:
            throw_invalid_type(self.node_type, self)

        for i, expr in enumerate(self.node_list):
            if (
                not is_expression(expr)
                and expr.node_type != NodeType.FUNCTION_DEFINITION
            ):
                throw_invalid_type(expr.node_type, self, f"expr#{i}")
