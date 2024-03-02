from ..node import Node, NodeType
from .statement import Statement
from .statement import is_statement
from ..expressions import Expression
from ..expressions import is_expression
from typing import List


class DoStatement(Statement):
    def __init__(self, node_type: NodeType, body: Statement, condition: Expression):
        super().__init__(node_type)
        self.condition = condition
        self.body = body
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.DO_STMT:
            raise TypeError(f"Invalid type: {self.node_type} for DoStatement")
        if not is_statement(self.body):
            raise TypeError(f"Invalid type {self.body.node_type} for do body")
        if not is_expression(self.condition):
            raise TypeError(f"Invalid type {self.condition.node_type} for do condition")

    def get_children(self) -> List[Node]:
        return [self.body, self.condition]

    def get_children_names(self) -> List[str]:
        return ["body", "condition"]
