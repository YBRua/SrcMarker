from ..node import Node, NodeType
from .expression import Expression
from .expression import is_primary_expression
from .expression_list import ExpressionList
from ..utils import throw_invalid_type
from typing import List


class CallExpression(Expression):
    def __init__(
        self,
        node_type: NodeType,
        callee: Expression,
        args: ExpressionList,
        optional: bool = False,
    ):
        super().__init__(node_type)
        self.callee = callee
        self.args = args
        self.optional = optional

    def _check_types(self):
        if self.node_type != NodeType.CALL_EXPR:
            throw_invalid_type(self.node_type, self)
        if not is_primary_expression(self.callee):
            throw_invalid_type(self.callee.node_type, self, "callee")
        if self.args.node_type != NodeType.EXPRESSION_LIST:
            throw_invalid_type(self.args.node_type, self, "args")

    def get_children(self) -> List[Node]:
        return [self.callee, self.args]

    def get_children_names(self) -> List[str]:
        return ["callee", "args"]
