from ..node import Node, NodeType
from .expression import Expression
from .expression import is_expression
from typing import List


class TernaryExpression(Expression):
    def __init__(
        self,
        node_type: NodeType,
        condition: Expression,
        consequence: Expression,
        alternate: Expression,
    ):
        super().__init__(node_type)
        self.condition = condition
        self.consequence = consequence
        self.alternative = alternate
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.TERNARY_EXPR:
            raise TypeError(f"Invalid type: {self.node_type} for TernaryExpression")
        if not is_expression(self.condition):
            raise TypeError(
                f"Invalid type: {self.condition.node_type} for ternary condition"
            )
        if not is_expression(self.consequence):
            raise TypeError(
                f"Invalid type: {self.consequence.node_type} for ternary consequence"
            )
        if not is_expression(self.alternative):
            raise TypeError(
                f"Invalid type: {self.alternative.node_type} for ternary alternative"
            )

    def get_children(self) -> List[Node]:
        return [self.condition, self.consequence, self.alternative]

    def get_children_names(self) -> List[str]:
        return ["condition", "consequence", "alternative"]
