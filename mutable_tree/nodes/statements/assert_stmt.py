from ..node import Node, NodeType
from .statement import Statement
from ..expressions import Expression
from ..expressions import is_expression
from typing import List, Optional


class AssertStatement(Statement):
    def __init__(
        self,
        node_type: NodeType,
        condition: Expression,
        message: Optional[Expression] = None,
    ):
        super().__init__(node_type)
        self.condition = condition
        self.message = message
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.ASSERT_STMT:
            raise TypeError(f"Invalid type: {self.node_type} for AssertStatement")
        if not is_expression(self.condition):
            raise TypeError(
                f"Invalid type: {self.condition.node_type} for assert condition"
            )
        if self.message is not None and not is_expression(self.message):
            raise TypeError(
                f"Invalid type: {self.message.node_type} for assert message"
            )

    def get_children(self) -> List[Node]:
        if self.message is not None:
            return [self.condition, self.message]
        else:
            return [self.condition]

    def get_children_names(self) -> List[str]:
        return ["condition", "message"]
